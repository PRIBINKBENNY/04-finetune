"""Build the dataset: messy job-post snippets -> clean JSON.

The task is structured extraction, chosen because it is easy to grade objectively (valid JSON, exact
fields) and small models are bad at it out of the box, so fine-tuning shows a clear before/after jump.

    python -m finetune.data --train 2000 --test 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIELDS = ["title", "company", "location", "remote", "salary_min", "salary_max", "currency", "skills"]

SYSTEM = (
    "Extract the job details from the text. Reply with ONLY a JSON object with keys: "
    "title (string), company (string), location (string or null), remote (true/false), "
    "salary_min (integer or null), salary_max (integer or null), currency (\"GBP\", \"USD\", \"EUR\", \"INR\" or null), "
    "skills (list of strings, lowercase)."
)

TITLES = ["Machine Learning Engineer", "Robotics Software Engineer", "Data Scientist", "AI Engineer",
          "Embedded Systems Engineer", "Computer Vision Engineer", "Backend Developer", "MLOps Engineer",
          "Graduate Software Engineer", "Junior Data Engineer", "Perception Engineer", "LLM Engineer"]
COMPANIES = ["Arcturus Labs", "Nimbus Robotics", "Quanta AI", "Helix Systems", "Bluefin Analytics",
             "Ostrich Dynamics", "Kite Mobility", "Lumen Health", "Vertex Logistics", "Sable Fintech",
             "Pinecrest Energy", "Orbital Farms"]
LOCATIONS = {"GBP": ["London", "Manchester", "Cambridge", "Bristol", "Edinburgh", "Hatfield"],
             "USD": ["San Francisco", "New York", "Austin", "Boston", "Seattle"],
             "EUR": ["Berlin", "Munich", "Amsterdam", "Dublin", "Paris"],
             "INR": ["Bengaluru", "Hyderabad", "Pune", "Chennai"]}
SYMBOL = {"GBP": "£", "USD": "$", "EUR": "€", "INR": "₹"}
SKILLS = ["python", "pytorch", "ros 2", "c++", "docker", "kubernetes", "aws", "sql", "opencv",
          "tensorflow", "linux", "rust", "fastapi", "spark", "llms", "rag", "embedded c", "git"]

TEMPLATES = [
    "{company} is hiring a {title} in {location}{remote_txt}. {salary_txt} You'll need {skills_txt}.",
    "Role: {title}\nCompany: {company}\nWhere: {location}{remote_txt}\nPay: {salary_txt}\nStack: {skills_txt}",
    "We're {company}! Looking for an awesome {title} ({location}{remote_txt}). Must know {skills_txt}. {salary_txt}",
    "[{company}] {title} - {location}{remote_txt} - {salary_txt} - skills: {skills_txt}",
    "Join {company} as our next {title}. Based {location_phrase}{remote_txt}. Experience with {skills_txt} is a must. {salary_txt}",
    "hiring!! {title} @ {company}. {remote_txt_lead}{skills_txt} people apply. {salary_txt}",
]


def _money(n: int, cur: str, style: int) -> str:
    sym = SYMBOL[cur]
    if cur == "INR":
        return f"{sym}{n // 100000} LPA" if style == 0 else f"INR {n:,}"
    if style == 0:
        return f"{sym}{n // 1000}k"
    if style == 1:
        return f"{sym}{n:,}"
    return f"{n:,} {cur}"


def make_example(rng: random.Random) -> dict:
    cur = rng.choice(list(LOCATIONS))
    remote = rng.random() < 0.35
    has_location = not (remote and rng.random() < 0.4)
    location = rng.choice(LOCATIONS[cur]) if has_location else None
    has_salary = rng.random() < 0.75
    if cur == "INR":
        lo = rng.randrange(6, 30) * 100000
        hi = lo + rng.randrange(2, 12) * 100000
    else:
        lo = rng.randrange(30, 140) * 1000
        hi = lo + rng.randrange(5, 40) * 1000
    single = has_salary and rng.random() < 0.2
    skills = rng.sample(SKILLS, rng.randint(2, 4))
    title, company = rng.choice(TITLES), rng.choice(COMPANIES)

    style = rng.randrange(3)
    if not has_salary:
        salary_txt = rng.choice(["Salary: competitive.", "Pay DOE.", ""])
    elif single:
        salary_txt = f"Up to {_money(hi, cur, style)}."
    else:
        salary_txt = f"{_money(lo, cur, style)} - {_money(hi, cur, style)}."
    remote_txt = rng.choice([" (remote OK)", ", fully remote", " / remote"]) if remote else ""
    skills_shown = [s.upper() if s in {"aws", "sql", "rag", "llms"} and rng.random() < 0.5 else s for s in skills]
    skills_txt = ", ".join(skills_shown[:-1]) + " and " + skills_shown[-1]
    text = rng.choice(TEMPLATES).format(
        company=company, title=title, location=location or "anywhere", remote_txt=remote_txt,
        remote_txt_lead="Remote. " if remote else "", salary_txt=salary_txt, skills_txt=skills_txt,
        location_phrase=f"in {location}" if location else "anywhere",
    ).strip()

    if location and location not in text:  # some templates omit the city: the label must match the text
        location = None
    target = {
        "title": title,
        "company": company,
        "location": location,
        "remote": remote,
        "salary_min": (None if single else lo) if has_salary else None,
        "salary_max": hi if has_salary else None,
        "currency": cur if has_salary else None,
        "skills": skills,
    }
    return {"text": text, "target": target}


def to_messages(example: dict, with_answer: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": example["text"]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": json.dumps(example["target"], ensure_ascii=False)})
    return msgs


def build(n_train: int, n_test: int, seed: int = 7) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n_train + n_test:
        ex = make_example(rng)
        if ex["text"] in seen:
            continue
        seen.add(ex["text"])
        rows.append(ex)
    return rows[:n_train], rows[n_train:]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    train, test = build(a.train, a.test, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("test", test)]:
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps({**r, "messages": to_messages(r)}, ensure_ascii=False) + "\n")
    print(f"wrote {len(train)} train / {len(test)} test examples to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
