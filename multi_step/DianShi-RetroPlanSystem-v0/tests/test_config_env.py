import textwrap

from runtime.config import _load_env_file, _resolve_env_value, load_config


def test_resolve_env_value_supports_all_reference_forms(monkeypatch):
    monkeypatch.setenv("BENCH_TEST_VAR", "hello")
    assert _resolve_env_value("$BENCH_TEST_VAR") == "hello"
    assert _resolve_env_value("${BENCH_TEST_VAR}") == "hello"
    assert _resolve_env_value("${BENCH_TEST_MISSING:-fallback}") == "fallback"
    assert _resolve_env_value("$BENCH_TEST_MISSING") == ""
    assert _resolve_env_value("plain-value") == "plain-value"


def test_load_config_expands_env_references(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_TEST_STEP_URL", "http://step:8001/v1/chat/completions")
    monkeypatch.setenv("BENCH_TEST_TEMPLATE", "/tmp/templates.jsonl")
    cfg_file = tmp_path / "lane.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            starting_molecules: "${BENCH_TEST_MISSING:-./data/stock.sqlite}"
            single_step:
              api_key: "$BENCH_TEST_KEY"
              template_filter:
                template_path: "$BENCH_TEST_TEMPLATE"
              retro:
                backend: "dianshi"
                api_url: "$BENCH_TEST_STEP_URL"
            """
        ).strip()
    )

    config = load_config(str(cfg_file))
    assert config["starting_molecules"] == "./data/stock.sqlite"
    assert config["single_step"]["api_key"] == ""
    assert config["single_step"]["template_filter"]["template_path"] == "/tmp/templates.jsonl"
    assert config["single_step"]["retro"]["api_url"] == "http://step:8001/v1/chat/completions"


def test_load_config_expands_model_names(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_TEST_PLANNER_MODEL", "planner-test")
    monkeypatch.setenv("BENCH_TEST_STEP_MODEL", "step-test")
    cfg_file = tmp_path / "lane.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            planner:
              default_model: "h-qwen/test"
            llm_config:
              providers:
                h-qwen:
                  llms:
                    test:
                      model_name: "$BENCH_TEST_PLANNER_MODEL"
            single_step:
              retro:
                backend: "dianshi"
                model: "$BENCH_TEST_STEP_MODEL"
            """
        ).strip()
    )

    config = load_config(str(cfg_file))
    llms = config["llm_config"]["providers"]["h-qwen"]["llms"]
    assert llms["test"]["model_name"] == "planner-test"
    assert config["single_step"]["retro"]["model"] == "step-test"


def test_load_env_file_does_not_override_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("BENCH_TEST_A=from-file\n# comment\nBENCH_TEST_B=also\n")
    monkeypatch.setenv("BENCH_TEST_A", "from-env")

    _load_env_file(env_file)

    import os

    assert os.environ["BENCH_TEST_A"] == "from-env"
    assert os.environ["BENCH_TEST_B"] == "also"
