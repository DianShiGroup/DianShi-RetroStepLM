#!/usr/bin/env python3
"""Unit tests for the Retro-R1 rdchiral template gate (extraction mocked)."""

from __future__ import annotations

import os
import sys
import itertools
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.gate.rdchiral_template_filter import (
    FilterResult,
    RdchiralTemplateFilter,
    _load_apply_library,
    build_rdchiral_template_filter,
)
from inference.gate.aot_reactant_ranker import AOTReactantRanker
from inference.stock import BuildingBlockChecker
from inference.step.dianshi_direct import (
    DianShiDirectStepRetroTool,
)

# Cache tables must stay in separate tables for RAG and no-RAG; the guard keys
# off the "no_rag" marker in the table name.

GATE_MODULE = "inference.gate.rdchiral_template_filter"
LIBRARY_IDS = itertools.count()
# Actual DianShi library rows 1, 2 and 7, ordered by training frequency.
ESTER_TEMPLATES = [
    "[O;D1;H0:3]=[C:2]-[OH;D1;+0:1]>>C-[O;H0;D2;+0:1]-[C:2]=[O;D1;H0:3]",
    "[O;D1;H0:3]=[C:2]-[OH;D1;+0:1]>>C-C-[O;H0;D2;+0:1]-[C:2]=[O;D1;H0:3]",
    "[O;D1;H0:3]=[C:2]-[OH;D1;+0:1]>>C-C(-C)(-C)-[O;H0;D2;+0:1]-[C:2]=[O;D1;H0:3]",
]
ESTER_LIBRARY = "\n".join(json.dumps({"key": smarts}) for smarts in ESTER_TEMPLATES)
ACID = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC"
TBU_ESTER = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)OC(C)(C)C)[C@@H](F)C(=O)OC"
METHYL_ESTER = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)OC)[C@@H](F)C(=O)OC"


class RdchiralTemplateFilterTests(unittest.TestCase):
    def _filter(
        self, library_line: str, match_mode: str = "exact", starting_mols=None, **extra: object
    ) -> RdchiralTemplateFilter:
        path = f"in-memory-library-{next(LIBRARY_IDS)}"
        config = {
            "backend": "rdchiral",
            "match_mode": match_mode,
            "template_path": str(path),
            "mapper_timeout": 5,
        }
        config.update(extra)
        # Exercise the real JSONL loaders without creating/deleting test files.
        with patch(f"{GATE_MODULE}._resolve_path", return_value=path), patch(
            f"{GATE_MODULE}.open", mock_open(read_data=library_line), create=True
        ):
            return RdchiralTemplateFilter(config, starting_mols=starting_mols)

    def test_pass_when_extracted_key_is_in_library(self):
        template_filter = self._filter('{"key": "[O:2].[C:1]>>[C:1]", "value": 2}\n')
        template_filter._extract_template = lambda _: "[O:2].[C:1]>>[C:1]"
        result = template_filter.validate(reactants=["C"], product="C")
        self.assertEqual(result.status, "PASS")

    def test_fail_when_key_missing(self):
        template_filter = self._filter('{"key": "[C:1]>>[C:1]", "value": 1}\n')
        template_filter._extract_template = lambda _: "[C:1]>>[N:1]"
        result = template_filter.validate(reactants=["C"], product="C")
        self.assertEqual(result.status, "FAIL")

    def test_empty_extraction_is_fail(self):
        template_filter = self._filter('{"key": "[C:1]>>[C:1]", "value": 1}\n')
        template_filter._extract_template = lambda _: ""
        result = template_filter.validate(reactants=["C"], product="C")
        self.assertEqual(result.status, "FAIL")

    def test_extraction_error_is_fail(self):
        template_filter = self._filter('{"key": "[C:1]>>[C:1]", "value": 1}\n')

        def _boom(_reaction_smiles: str) -> str:
            raise RuntimeError("extract exploded")

        template_filter._extract_template = _boom
        result = template_filter.validate(reactants=["C"], product="C")
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.detail, "RuntimeError")

    def test_disabled_config_builds_nothing(self):
        self.assertIsNone(build_rdchiral_template_filter(None))
        self.assertIsNone(build_rdchiral_template_filter({"enabled": False}))

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            RdchiralTemplateFilter({"backend": "localtemplate", "template_path": "x"})

    def test_apply_pass_when_both_sides_match_and_template_applies(self):
        template_filter = self._filter(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            match_mode="apply",
        )
        result = template_filter.validate(reactants=["C", "O"], product="CO")
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.detail, "apply-match")
        self.assertEqual(set(result.reactants or ()), {"C", "O"})

    def test_apply_fail_when_no_template_matches(self):
        template_filter = self._filter(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            match_mode="apply",
            blurry=False,
        )
        result = template_filter.validate(reactants=["N"], product="c1ccccc1")
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.detail, "no template matches reaction")

    def test_apply_fail_when_matched_template_does_not_apply(self):
        template_filter = self._filter(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            match_mode="apply",
            blurry=False,
        )
        template_filter._run_retro_outputs = lambda product, smarts: []
        result = template_filter.validate(reactants=["C", "O"], product="CO")
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.detail, "matched template does not apply")

    def test_blurry_rewrites_reactants_when_first_pass_misses(self):
        template_filter = self._filter(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            match_mode="apply",
        )
        result = template_filter.validate(reactants=["N"], product="CO")
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.detail, "blurry")
        self.assertEqual(set(result.reactants or ()), {"C", "O"})

    def test_blurry_fail_when_no_template_applies(self):
        template_filter = self._filter(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            match_mode="apply",
        )
        result = template_filter.validate(reactants=["N"], product="c1ccccc1")
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.detail, "blurry search found no template")

    def test_non_pass_should_be_dropped_by_caller(self):
        dropped = [
            FilterResult("FAIL", "template_filter", "template is absent from library"),
            FilterResult("UNKNOWN", "template_filter", "invalid canonical input"),
        ]
        kept = FilterResult("PASS", "template_filter")
        self.assertTrue(all(item.status != "PASS" for item in dropped))
        self.assertEqual(kept.status, "PASS")

    def test_later_matching_template_wins_when_its_output_is_in_stock(self):
        gate = self._filter(ESTER_LIBRARY, "apply", starting_mols={TBU_ESTER}, blurry=False)
        result = gate.validate(reactants=[TBU_ESTER], product=ACID)
        self.assertEqual(result.reactants, (TBU_ESTER,))
        self.assertEqual(result.rank_score, 0.0)
        self.assertEqual(result.detail, "apply-match")

    def test_ranking_none_preserves_first_hit_auto_fix(self):
        gate = self._filter(
            ESTER_LIBRARY, "apply", starting_mols={TBU_ESTER}, ranking="none", blurry=False
        )
        result = gate.validate(reactants=[TBU_ESTER], product=ACID)
        self.assertEqual(result.reactants, (METHYL_ESTER,))
        self.assertIsNone(result.rank_score)

    def test_apply_template_limit_keeps_first_hit(self):
        gate = self._filter(
            ESTER_LIBRARY,
            "apply",
            starting_mols={TBU_ESTER},
            apply_max_templates=1,
            blurry=False,
        )
        result = gate.validate(reactants=[TBU_ESTER], product=ACID)
        self.assertEqual(result.reactants, (METHYL_ESTER,))

    def test_blurry_ranks_across_templates_and_still_rewrites(self):
        gate = self._filter(ESTER_LIBRARY, "apply", starting_mols={TBU_ESTER})
        result = gate.validate(reactants=["N"], product=ACID)
        self.assertEqual(result.reactants, (TBU_ESTER,))
        self.assertEqual(result.detail, "blurry")

    def test_empty_first_matching_template_still_uses_later_apply_hit(self):
        gate = self._filter(ESTER_LIBRARY, "apply", starting_mols={TBU_ESTER}, blurry=False)

        def _run(product, smarts):
            if smarts == ESTER_TEMPLATES[0]:
                return []
            return RdchiralTemplateFilter._run_retro_outputs(gate, product, smarts)

        gate._run_retro_outputs = _run
        result = gate.validate(reactants=[TBU_ESTER], product=ACID)
        self.assertEqual(result.detail, "apply-match")
        self.assertEqual(result.reactants, (TBU_ESTER,))

    def test_product_in_generated_reactants_is_fail(self):
        gate = self._filter(
            '{"key": "[C:1]>>[C:1]", "value": 1}\n',
            match_mode="apply",
            ranking="none",
            blurry=True,
        )
        gate._run_retro_outputs = Mock(return_value=[["C"]])
        result = gate.validate(reactants=["C"], product="C")
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.detail, "product overlaps reactants")

    def test_blurry_template_limit_is_respected(self):
        gate = self._filter(ESTER_LIBRARY, "apply", ranking="none", blurry_top_n=1)
        library = _load_apply_library(gate.template_path)
        ranked = [(1.0, 0), (1.0, 1), (1.0, 2)]
        order = gate._blurry_ranked_indices("N", library, ranked)
        self.assertEqual(len(order), 1)

    def test_blurry_reranks_pool_by_reactant_similarity(self):
        library_line = "\n".join(
            json.dumps({"key": smarts})
            for smarts in ("[C:1][O:2]>>[F:1]", "[C:1][O:2]>>[N:1]")
        )
        gate = self._filter(library_line, "apply", ranking="none")
        library = _load_apply_library(gate.template_path)
        order = gate._blurry_ranked_indices("N", library, [(1.0, 0), (1.0, 1)])
        self.assertEqual(order[0], 1)

    def test_ranks_all_outputs_and_skips_invalid_splits(self):
        gate = self._filter(ESTER_LIBRARY, "apply", starting_mols={"CC"})
        gate._run_retro_outputs = Mock(return_value=[["invalid"], ["CCC"], ["CC"]])
        result = gate.validate(reactants=[TBU_ESTER], product=ACID)
        self.assertEqual(result.reactants, ("CC",))
        self.assertEqual(result.rank_score, 0.0)

    def test_lower_scscore_wins_without_stock(self):
        gate = self._filter(ESTER_LIBRARY, "apply")
        gate._run_retro_outputs = Mock(return_value=[["CCC"], ["CC"]])
        gate.ranker._molecule_score = lambda smi: {"CCC": 4.0, "CC": 2.0}[smi]
        result = gate.validate(reactants=[TBU_ESTER], product=ACID)
        self.assertEqual(result.reactants, ("CC",))
        self.assertEqual(result.rank_score, -4.0)

    def test_ties_preserve_order_and_duplicate_splits_are_scored_once(self):
        gate = self._filter(ESTER_LIBRARY, "apply")
        gate._run_retro_outputs = Mock(return_value=[["CCC"], ["CC"], ["CCC"]])
        gate.ranker.score = Mock(return_value=-4.0)
        result = gate.validate(reactants=[TBU_ESTER], product=ACID)
        self.assertEqual(result.reactants, ("CCC",))
        self.assertEqual(gate.ranker.score.call_count, 2)

    def test_aot_reward_ignores_stock_and_penalizes_unsolved_count(self):
        ranker = AOTReactantRanker({"O"})
        ranker._molecule_score = lambda smi: {"CC": 2.0, "CCC": 4.0}[smi]
        self.assertEqual(ranker.score(("O",)), 0.0)
        self.assertEqual(ranker.score(("O", "CC", "CCC")), -9.0)
        self.assertAlmostEqual(ranker.score(("O", "CC", "CCC"), 0.1), -10.8)

    def test_stock_is_passed_from_single_step_tool_to_gate(self):
        with patch.object(DianShiDirectStepRetroTool, "_load_starting_molecules", return_value={"CC"}), patch(
            "inference.step.dianshi_direct.build_rdchiral_template_filter"
        ) as build:
            config = {"enabled": True}
            tool = DianShiDirectStepRetroTool(template_filter=config)
        build.assert_called_once_with(config, starting_mols=tool.starting_mols)

    def test_tool_sorts_after_rewrite_and_dedup_before_topk(self):
        with patch.object(DianShiDirectStepRetroTool, "_load_starting_molecules", return_value={"CC"}):
            tool = DianShiDirectStepRetroTool(topk=1)
        tool._call_api = Mock(
            return_value=[
                {"rank": 1, "output_smiles": "CCC", "parsed": ["CCC"]},
                {"rank": 2, "output_smiles": "CCCC", "parsed": ["CCCC"]},
                {"rank": 3, "output_smiles": "CCCCC", "parsed": ["CCCCC"]},
            ]
        )
        tool.template_filter = Mock()
        filtered = [
            FilterResult("PASS", "template_filter", reactants=("CC",), rank_score=0),
            FilterResult("PASS", "template_filter", reactants=("CCC",), rank_score=-8),
        ]
        tool.template_filter.filter_candidates.side_effect = [filtered, filtered]
        result = tool.execute({"molecule": "CCO"})
        self.assertEqual(result["results"], [[["CC", "available"]]])
        second = tool.execute({"molecule": "CCO", "topk": 3})
        self.assertEqual(second["results"], [[["CC", "available"]], [["CCC", "unavailable"]]])
        self.assertEqual(tool._call_api.call_count, 2)

    def test_disabled_gate_keeps_model_order(self):
        with patch.object(DianShiDirectStepRetroTool, "_load_starting_molecules", return_value=set()):
            tool = DianShiDirectStepRetroTool(topk=1)
        tool._call_api = Mock(
            return_value=[
                {"rank": 1, "output_smiles": "CCC", "parsed": ["CCC"]},
                {"rank": 2, "output_smiles": "CC", "parsed": ["CC"]},
            ]
        )
        self.assertEqual(tool.execute({"molecule": "CCO"})["results"], [[["CCC", "unavailable"]]])





    def test_filter_candidates_is_the_batch_entry(self):
        gate = self._filter(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            match_mode="apply",
        )
        kept = gate.filter_candidates(product="CO", candidates=[["C", "O"], ["N"]])
        self.assertEqual(len(kept), 1)
        self.assertEqual(set(kept[0].reactants or ()), {"C", "O"})

    def test_sqlite_stock_canonicalizes_before_lookup(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = str(Path(temp_dir.name) / "stock.sqlite")
        canon = (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4CC(Cl)"
            "CC[C@]4(C)[C@H]3CC[C@]12C"
        )
        raw = (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4CC(Cl)"
            "CC[C@]4(C)[C@H]3CC[C@@]21C"
        )
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE stocks(smiles TEXT PRIMARY KEY) WITHOUT ROWID")
        con.execute("INSERT INTO stocks VALUES (?)", (canon,))
        con.commit()
        con.close()
        stock = BuildingBlockChecker(path)
        self.assertIn(raw, stock)
        self.assertIn(canon, stock)
        self.assertNotIn("CCO", stock)
        self.assertEqual(len(stock), 1)
        ranker = AOTReactantRanker(stock)
        ranker._molecule_score = lambda smi: 2.0
        self.assertEqual(ranker.score((raw,)), 0.0)
        self.assertEqual(ranker.score(("CCO",)), -4.0)

    def test_filter_candidates_parallel_matches_serial(self):
        gate = self._filter(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            match_mode="apply",
            ranking="none",
        )
        cands = [["N"], ["C", "O"], ["CC"]]
        os.environ["AOT_GATE_CAND_WORKERS"] = "1"
        serial = [
            tuple(sorted(item.reactants or ()))
            for item in gate.filter_candidates(product="CO", candidates=cands)
        ]
        os.environ["AOT_GATE_CAND_WORKERS"] = "10"
        parallel = [
            tuple(sorted(item.reactants or ()))
            for item in gate.filter_candidates(product="CO", candidates=cands)
        ]
        os.environ.pop("AOT_GATE_CAND_WORKERS", None)
        self.assertEqual(serial, parallel)
        self.assertEqual(serial, [("C", "O")])

    def test_template_parallel_matches_serial(self):
        gate = self._filter(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n' * 5,
            match_mode="apply",
            ranking="none",
        )
        cands = [["N"], ["C", "O"]]
        os.environ["AOT_GATE_TMPL_WORKERS"] = "1"
        serial = [
            tuple(sorted(item.reactants or ()))
            for item in gate.filter_candidates(product="CO", candidates=cands)
        ]
        os.environ["AOT_GATE_TMPL_WORKERS"] = "4"
        parallel = [
            tuple(sorted(item.reactants or ()))
            for item in gate.filter_candidates(product="CO", candidates=cands)
        ]
        os.environ.pop("AOT_GATE_TMPL_WORKERS", None)
        self.assertEqual(serial, parallel)
        self.assertEqual(serial, [("C", "O")])



if __name__ == "__main__":
    unittest.main()
