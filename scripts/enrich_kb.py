#!/usr/bin/env python3
"""Enrich exercise_kb.json with muscle data from wger (wger.de/api/v2).

Covers two cases:
  1. Exercises already in the KB missing muscle data
  2. Unresolved exercise names in the DB not yet in the KB

Run manually after ingestion to grow and enrich the KB.
Strategy: fetch all exercises from wger once, filter client-side (server-side
filtering is non-functional on wger's public instance).
"""
import json
import re
import time

import requests
from rapidfuzz import process as fuzz_process

from liftaudit.dbstore.connection import DEFAULT_DB_PATH, connect_readonly
from liftaudit.ingestion.exercise_resolver import KB_PATH

WGER_BASE = "https://wger.de/api/v2"
LANGUAGE_ENGLISH = 2
FUZZY_THRESHOLD = 80


def fetch_all_exercises() -> list[dict]:
    """Fetch all exercises from wger exerciseinfo, paginating as needed."""
    exercises = []
    url = f"{WGER_BASE}/exerciseinfo/"
    params = {"format": "json", "limit": 100, "offset": 0}
    while url:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        exercises.extend(data["results"])
        url = data.get("next")
        params = {}
        time.sleep(0.3)
    return exercises


def build_index(exercises: list[dict]) -> dict[str, dict]:
    """Build a lowercase name → exercise dict using English translations."""
    index = {}
    for ex in exercises:
        for t in ex.get("translations", []):
            if t.get("language") == LANGUAGE_ENGLISH and t.get("name"):
                index[t["name"].lower()] = ex
    return index


def find_best_match(query: str, index: dict[str, dict]) -> tuple[str, dict] | None:
    """Find the best matching exercise name using fuzzy matching."""
    result = fuzz_process.extractOne(
        query.lower(), index.keys(), score_cutoff=FUZZY_THRESHOLD
    )
    if result:
        matched_name, _score, _ = result
        return matched_name, index[matched_name]
    return None


def muscles_from_exercise(ex: dict) -> dict:
    return {
        "target_muscles": [m["name_en"] for m in ex.get("muscles", []) if m.get("name_en")],
        "secondary_muscles": [m["name_en"] for m in ex.get("muscles_secondary", []) if m.get("name_en")],
        "body_parts": [ex["category"]["name"]] if ex.get("category") else [],
    }


def load_kb() -> dict:
    with KB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_kb(kb: dict) -> None:
    with KB_PATH.open("w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)
    print(f"Saved {KB_PATH}")


def unresolved_from_db() -> list[str]:
    with connect_readonly(DEFAULT_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT input_name FROM unresolved_exercises"
        ).fetchall()
    return [r[0] for r in rows]


def main():
    print("Fetching all exercises from wger...")
    all_exercises = fetch_all_exercises()
    index = build_index(all_exercises)
    print(f"  {len(all_exercises)} exercises indexed ({len(index)} English names)")

    kb = load_kb()
    exercises = kb["exercises"]
    canonical_names = {e["canonical_name"] for e in exercises}
    updated = False

    # --- Case 1: enrich existing KB entries missing muscle data ---
    for exercise in exercises:
        if exercise.get("target_muscles"):
            continue
        name = exercise["display_name"]
        print(f"Matching: {name}")
        match = find_best_match(name, index)
        if match:
            matched_name, ex = match
            muscles = muscles_from_exercise(ex)
            exercise.update(muscles)
            print(f"  -> matched '{matched_name}' | targets: {exercise['target_muscles']}")
            updated = True
        else:
            print(f"  -> no match")

    # --- Case 2: add new entries for unresolved DB exercises ---
    for input_name in unresolved_from_db():
        print(f"Looking up unresolved: {input_name}")
        match = find_best_match(input_name, index)
        if not match:
            print(f"  -> no match")
            continue

        matched_name, ex = match
        canonical = re.sub(r"[^a-z0-9]+", "_", matched_name).strip("_")

        if canonical in canonical_names:
            print(f"  -> already in KB as '{canonical}', adding alias")
            for e in exercises:
                if e["canonical_name"] == canonical and input_name not in e.get("aliases", []):
                    e.setdefault("aliases", []).append(input_name)
                    updated = True
        else:
            entry = {
                "canonical_name": canonical,
                "display_name": matched_name.title(),
                "aliases": [input_name],
                **muscles_from_exercise(ex),
            }
            exercises.append(entry)
            canonical_names.add(canonical)
            print(f"  -> added '{canonical}' | targets: {entry['target_muscles']}")
            updated = True

    if updated:
        save_kb(kb)
    else:
        print("Nothing to update.")


if __name__ == "__main__":
    main()
