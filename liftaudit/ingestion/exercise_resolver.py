import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from liftaudit.ingestion.schemas import ExerciseResolution

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only when rapidfuzz is absent
    fuzz = None


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
            )

        return ExerciseResolution(input_name=exercise_name, status="unresolved")

    def _load_kb(self, path: Path) -> List[Dict[str, object]]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data["exercises"]

    def _build_lookup(self) -> Dict[str, Tuple[str, str, str]]:
        lookup: Dict[str, Tuple[str, str, str]] = {}
        for exercise in self.exercises:
            canonical = str(exercise["canonical_name"])
            lookup[normalize_name(canonical)] = (canonical, canonical, "exact")
            for alias in exercise.get("aliases", []):
                alias_text = str(alias)
                lookup[normalize_name(alias_text)] = (canonical, alias_text, "alias")
        return lookup

    def _best_fuzzy_match(self, normalized: str) -> Optional[Tuple[str, str, float]]:
        best: Optional[Tuple[str, str, float]] = None
        for candidate, (canonical, matched_alias, _status) in self._exact_lookup.items():
            score = self._similarity(normalized, candidate)
            if score >= self.fuzzy_threshold and (best is None or score > best[2]):
                best = (canonical, matched_alias, score)
        return best

    def _similarity(self, left: str, right: str) -> float:
        if fuzz is not None:
            return float(fuzz.token_sort_ratio(left, right))
        return _difflib_ratio(left, right)


def _difflib_ratio(left: str, right: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio() * 100


def resolve_exercise_names(names: Iterable[str]) -> List[ExerciseResolution]:
    resolver = ExerciseResolver()
    return [resolver.resolve(name) for name in names]

