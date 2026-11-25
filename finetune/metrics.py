"""Grade model outputs against the gold JSON."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .data import FIELDS

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_json(text: str) -> dict | None:
    """Accept a bare object, a fenced block, or an object embedded in chatter."""
    t = _FENCE.sub("", text.strip())
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _norm(field_name: str, v):
    if field_name == "skills":
        return sorted({str(s).strip().lower() for s in v}) if isinstance(v, list) else None
    if isinstance(v, str):
        s = v.strip()
        if field_name in {"salary_min", "salary_max"}:
            digits = re.sub(r"[^\d]", "", s)
            return int(digits) if digits else None
        if field_name == "remote":
            return s.lower() in {"true", "yes", "1"}
        return s.lower() if s.lower() not in {"", "null", "none", "n/a"} else None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


@dataclass
class Report:
    n: int = 0
    valid_json: int = 0
    exact: int = 0
    field_correct: dict = field(default_factory=lambda: {f: 0 for f in FIELDS})
    skills_tp: int = 0
    skills_fp: int = 0
    skills_fn: int = 0

    def add(self, pred_text: str, gold: dict) -> dict:
        self.n += 1
        pred = parse_json(pred_text)
        row = {"valid_json": pred is not None, "fields": {}}
        if pred is None:
            self.skills_fn += len(gold["skills"])
            return row
        self.valid_json += 1
        all_ok = True
        for f in FIELDS:
            ok = _norm(f, pred.get(f)) == _norm(f, gold.get(f))
            row["fields"][f] = ok
            self.field_correct[f] += ok
            all_ok &= ok
        p, g = set(_norm("skills", pred.get("skills")) or []), set(_norm("skills", gold["skills"]))
        self.skills_tp += len(p & g)
        self.skills_fp += len(p - g)
        self.skills_fn += len(g - p)
        self.exact += all_ok
        row["exact"] = all_ok
        return row

    def summary(self) -> dict:
        pct = lambda x: round(100 * x / self.n, 1) if self.n else 0.0  # noqa: E731
        prec = self.skills_tp / max(self.skills_tp + self.skills_fp, 1)
        rec = self.skills_tp / max(self.skills_tp + self.skills_fn, 1)
        return {
            "examples": self.n,
            "valid_json_pct": pct(self.valid_json),
            "exact_match_pct": pct(self.exact),
            "field_accuracy_pct": round(sum(pct(v) for v in self.field_correct.values()) / len(FIELDS), 1),
            "per_field_pct": {f: pct(v) for f, v in self.field_correct.items()},
            "skills_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
        }
