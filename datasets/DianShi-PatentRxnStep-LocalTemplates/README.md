# DianShi-PatentRxnStep-LocalTemplates

Local reaction templates (LocalTemplate) of `DianShi-PatentRxnStep-Train-Full`,
and the template-coverage partitions used in the template-restriction
experiments with seeds 42, 43 and 44.

## Files

| File | Rows | Content |
|---|---:|---|
| `DianShi-PatentRxnStep-Train-Full_local_templates.jsonl` | 31,855 | Every local template in Train-Full |
| `DianShi-PatentRxnStep-Train-TemplateRestricted-S42_local_templates.jsonl` | 30,935 | Templates kept in the seed-42 training set |
| `DianShi-PatentRxnStep-Train-TemplateRestricted-S43_local_templates.jsonl` | 30,935 | Same for seed 43 |
| `DianShi-PatentRxnStep-Train-TemplateRestricted-S44_local_templates.jsonl` | 30,935 | Same for seed 44 |
| `DianShi-PatentRxnStep-Test-TemplatePartition-S42_local_templates.jsonl` | 1,841 | Templates of the two test sets, each marked `retained` or `withheld` for seed 42 |
| `DianShi-PatentRxnStep-Test-TemplatePartition-S43_local_templates.jsonl` | 1,841 | Same for seed 43 |
| `DianShi-PatentRxnStep-Test-TemplatePartition-S44_local_templates.jsonl` | 1,841 | Same for seed 44 |

## Formats

One template per line. Training libraries:

```json
{"local_template": "<local template>"}
```

Test partitions:

```json
{"local_template": "<local template>", "status": "retained" | "withheld"}
```

The test partitions list the templates of Test-2001-2024 and Test-2025
together (`../DianShi-PatentRxnStep-Dataset/`); a template can occur in one test
set or in both. A test reaction's template is its `localtemp_res.rxn_temp`
field, so a reaction is retained or withheld by looking that field up here.

## How the partition works

For each seed, about half of the distinct templates in the two test sets
(920 of 1,841) are `withheld`, chosen by a seed-specific SHA-256 ranking. A
template shared by both test sets gets one decision. Every training reaction
with a withheld template is removed from Train-Full; the rest form that seed's
`TemplateRestricted` training set. So:

- no `withheld` template appears in that seed's training library;
- every `retained` template does, with all of its Train-Full reactions kept;
- the 30,014 Train-Full templates that occur in neither test set are always kept.

Train-Full has 2,712,259 reactions. The template-restricted training sets keep
1,804,926 (seed 42), 1,673,416 (seed 43) and 1,564,454 (seed 44).

Test reactions per subset (retained / withheld):

| Seed | Test-2001-2024 | Test-2025 |
|---|---|---|
| 42 | 1,977 / 1,821 | 970 / 916 |
| 43 | 1,842 / 1,956 | 937 / 949 |
| 44 | 1,797 / 2,001 | 878 / 1,008 |

Templates per subset are 671 / 671 for Test-2001-2024 and 426 / 427 for
Test-2025 in all three seeds. The seeds differ in which templates fall on which
side: 884–918 of the 1,841 test templates change status between any two seeds,
and only 246 are retained in all three.

## Consistency

The files were extracted from the full training records, and each training
record has exactly one template. For every seed, the kept and removed training
reactions add back up to Train-Full template by template, and the removed
reactions carry exactly the 920 `withheld` templates.

## LocalTemplate vs rdchiral

These are **LocalTemplate** templates, used to define template coverage. The
library used for reaction validation in multi-step planning is the separate
**rdchiral** library (412,027 templates) in `DianShi-RetroPlanSystem-Assets`.
Both are extracted from the same 2,712,259 Train-Full reactions.
