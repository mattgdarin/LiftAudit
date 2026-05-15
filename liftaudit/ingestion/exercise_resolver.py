import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from liftaudit.ingestion.schemas import ExerciseResolution
from rapidfuzz import fuzz


KB_PATH = Path(__file__).resolve().parents[1] / "data" / "exercise_kb.json"


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


class ExerciseResolver:
    def __init__(self, kb_path: Optional[Path] = None, fuzzy_threshold: float = 86.0):
        self.kb_path = kb_path or KB_PATH
        self.fuzzy_threshold = fuzzy_threshold
        self.exercises = self._load_kb(self.kb_path)
        self._exact_lookup = self._build_lookup()

    def resolve(self, exercise_name: str) -> ExerciseResolution:
        normalized = normalize_name(exercise_name)
        if not normalized:
            return ExerciseResolution(input_name=exercise_name, status="unresolved")

        exact = self._exact_lookup.get(normalized)
        if exact:
            canonical, matched_alias, status = exact
            return ExerciseResolution(
                input_name=exercise_name,
                canonical_name=canonical,
                status=status,
                matched_alias=matched_alias,
                score=100.0,
                **self._muscle_data(canonical),
            )

        fuzzy_match = self._best_fuzzy_match(normalized)
        if fuzzy_match:
            canonical, matched_alias, score = fuzzy_match
            return ExerciseResolution(
                input_name=exercise_name,
                canonical_name=canonical,
                status="fuzzy",
                matched_alias=matched_alias,
                score=score,
                **self._muscle_data(canonical),
            )

        return ExerciseResolution(input_name=exercise_name, status="unresolved")

    def _load_kb(self, path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)["exercises"]

    def _build_lookup(self) -> Dict[str, Tuple[str, str, str]]:
        lookup: Dict[str, Tuple[str, str, str]] = {}
        for exercise in self.exercises:
            canonical = str(exercise["canonical_name"])
            lookup[normalize_name(canonical)] = (canonical, canonical, "exact")
            for alias in exercise.get("aliases", []):
                lookup[normalize_name(str(alias))] = (canonical, str(alias), "alias")
        return lookup

    def _muscle_data(self, canonical_name: str) -> Dict[str, List[str]]:
        for exercise in self.exercises:
            if exercise["canonical_name"] == canonical_name:
                return {
                    "target_muscles": exercise.get("target_muscles", []),
                    "secondary_muscles": exercise.get("secondary_muscles", []),
                    "body_parts": exercise.get("body_parts", []),
                }
        return {"target_muscles": [], "secondary_muscles": [], "body_parts": []}

    def _best_fuzzy_match(self, normalized: str) -> Optional[Tuple[str, str, float]]:
        best: Optional[Tuple[str, str, float]] = None
        for candidate, (canonical, matched_alias, _) in self._exact_lookup.items():
            score = float(fuzz.token_sort_ratio(normalized, candidate))
            if score >= self.fuzzy_threshold and (best is None or score > best[2]):
                best = (canonical, matched_alias, score)
        return best


def resolve_exercise_names(names: Iterable[str]) -> List[ExerciseResolution]:
    resolver = ExerciseResolver()
    return [resolver.resolve(name) for name in names]
