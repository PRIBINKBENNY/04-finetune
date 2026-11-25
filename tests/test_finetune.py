import json
import re

from finetune import data
from finetune.evaluate import compare, run
from finetune.metrics import Report, parse_json
from finetune.train import Config


def test_dataset_is_deterministic_unique_and_consistent():
    tr1, te1 = data.build(300, 50, seed=1)
    tr2, _ = data.build(300, 50, seed=1)
    assert [r["text"] for r in tr1] == [r["text"] for r in tr2]
    texts = [r["text"] for r in tr1 + te1]
    assert len(set(texts)) == len(texts)
    for r in tr1:
        t, g = r["text"], r["target"]
        assert g["company"] in t and g["title"] in t
        assert set(g) == set(data.FIELDS)
        assert all(s.lower() in t.lower() for s in g["skills"])
        if g["location"]:
            assert g["location"] in t
        assert g["remote"] == ("remote" in t.lower())
        if g["salary_max"] is None:
            assert g["currency"] is None and g["salary_min"] is None


def test_messages_format():
    ex = data.build(1, 0)[0][0]
    m = data.to_messages(ex)
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert json.loads(m[-1]["content"]) == ex["target"]
    assert data.to_messages(ex, with_answer=False)[-1]["role"] == "user"


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}
    assert parse_json("not json") is None
    assert parse_json("[1,2]") is None


def test_metrics_perfect_partial_and_broken():
    gold = {"title": "AI Engineer", "company": "Quanta AI", "location": "London", "remote": False,
            "salary_min": 50000, "salary_max": 60000, "currency": "GBP", "skills": ["python", "rag"]}
    rep = Report()
    assert rep.add(json.dumps(gold), gold)["exact"]
    # normalisation: case, string numbers, skill order
    loose = {**gold, "title": "ai engineer", "salary_min": "50,000", "skills": ["RAG", "Python"]}
    assert rep.add(json.dumps(loose), gold)["exact"]
    wrong = {**gold, "salary_max": 65000, "skills": ["python", "sql"]}
    row = rep.add(json.dumps(wrong), gold)
    assert not row["exact"] and row["fields"]["salary_max"] is False and row["fields"]["company"]
    assert not rep.add("I think it's an AI job", gold)["valid_json"]
    s = rep.summary()
    assert s["examples"] == 4 and s["valid_json_pct"] == 75.0 and s["exact_match_pct"] == 50.0
    assert 0 < s["skills_f1"] < 1


def test_eval_run_with_oracle_and_naive_predictors():
    _, test = data.build(0, 40, seed=3)
    by_text = {r["text"]: r["target"] for r in test}

    def oracle(batch):
        return [json.dumps(by_text[m[-1]["content"]]) for m in batch]

    def naive(batch):  # what an untuned small model often does: chatty, partial
        out = []
        for m in batch:
            company = re.search(r"(\w+ \w+)", m[-1]["content"]).group(1)
            out.append(f'Here is the JSON: {{"company": "{company}", "remote": "maybe"}}')
        return out

    s_good, details = run(oracle, test)
    s_bad, _ = run(naive, test)
    assert s_good["exact_match_pct"] == 100.0 and len(details) == 40
    assert s_bad["exact_match_pct"] == 0.0 and s_bad["valid_json_pct"] == 100.0


def test_compare_table(tmp_path):
    for tag, em in [("base", 4.5), ("finetuned", 91.0)]:
        (tmp_path / f"{tag}.json").write_text(json.dumps({"tag": tag, "summary": {
            "valid_json_pct": 80, "exact_match_pct": em, "field_accuracy_pct": 70, "skills_f1": 0.8}}))
    table = compare([str(tmp_path / "base.json"), str(tmp_path / "finetuned.json")])
    assert "| finetuned | 80 | 91.0 |" in table


def test_config_and_dry_run(tmp_path, capsys):
    from finetune import train as tr

    cfg = Config.load("configs/qlora_t4.yaml")
    assert cfg.load_in_4bit and cfg.lora.r == 16 and "q_proj" in cfg.lora.target_modules
    plan = cfg.plan(2000)
    assert plan["effective_batch"] == 16 and plan["optimizer_steps"] == 250
    assert tr.main(["--config", "configs/qlora_t4.yaml", "--dry-run"]) == 0
    assert "optimizer_steps" in capsys.readouterr().out
