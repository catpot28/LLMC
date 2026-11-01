"""Phase 3 building blocks for boilerplate and duplicate removal.

Feature 3.1 segments retained page text into blocks and creates a separate,
normalized representation for comparison. It does not remove page content.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


BLANK_LINE_RE = re.compile(r"\n[\t \f\v]*\n+")
WHITESPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")

# Feature 3.2 thresholds. A block must recur across both an absolute number and a
# proportion of company pages before its position or sequence is considered.
MIN_RECURRING_PAGES = 3
MIN_RECURRING_PAGE_FRACTION = 0.20
EDGE_PAGE_FRACTION = 0.20
MIN_EDGE_OCCURRENCE_FRACTION = 0.80
MIN_RECURRING_RUN_LENGTH = 3
MAX_EDGE_BLOCK_CHARACTERS = 2_000
MAX_INTERIOR_BLOCK_CHARACTERS = 300

# Feature 3.4 thresholds. Five-word shingles preserve local word order while
# tolerating a small number of edits elsewhere in an otherwise duplicate page.
NEAR_DUPLICATE_SHINGLE_SIZE = 5
NEAR_DUPLICATE_MIN_WORDS = 50
NEAR_DUPLICATE_MIN_SHINGLES = 20
NEAR_DUPLICATE_MIN_LENGTH_RATIO = 0.90
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.90
NEAR_DUPLICATE_ANCHORS_PER_PAGE = 32
NEAR_DUPLICATE_MAX_ANCHOR_PAGE_FRACTION = 0.10
WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class TextBlock:
    """One ordered piece of page text and its comparison representation."""

    index: int
    text: str
    normalized: str
    character_count: int
    normalized_character_count: int


@dataclass(frozen=True)
class RecurringBlockStats:
    """Company-level recurrence and position evidence for one normalized block."""

    normalized: str
    page_count: int
    page_fraction: float
    occurrence_count: int
    edge_occurrence_count: int
    edge_occurrence_fraction: float


@dataclass(frozen=True)
class BoilerplateDecision:
    """Auditable decision to remove one occurrence of a recurring block."""

    page_index: int
    block_index: int
    text: str
    normalized: str
    reasons: tuple[str, ...]
    recurring_page_count: int
    recurring_page_fraction: float
    edge_occurrence_fraction: float


@dataclass(frozen=True)
class ExactDuplicateGroup:
    """One normalized-exact page group and its selected canonical member."""

    fingerprint: str
    canonical_page_index: int
    member_page_indexes: tuple[int, ...]


@dataclass(frozen=True)
class ExactDuplicateDecision:
    """Auditable decision identifying a non-canonical exact duplicate page."""

    page_index: int
    canonical_page_index: int
    fingerprint: str
    group_size: int
    reason: str = "normalized_exact_duplicate"


@dataclass(frozen=True)
class ExactDuplicateResult:
    """Canonical indexes, duplicate decisions, and complete duplicate groups."""

    canonical_page_indexes: tuple[int, ...]
    decisions: tuple[ExactDuplicateDecision, ...]
    groups: tuple[ExactDuplicateGroup, ...]


@dataclass(frozen=True)
class NearDuplicateDecision:
    """Auditable decision linking a near duplicate to a canonical page."""

    page_index: int
    canonical_page_index: int
    jaccard_similarity: float
    length_ratio: float
    shared_shingles: int
    page_shingles: int
    canonical_shingles: int
    reason: str = "near_duplicate"


@dataclass(frozen=True)
class NearDuplicateResult:
    """Canonical indexes and non-canonical near-duplicate decisions."""

    canonical_page_indexes: tuple[int, ...]
    decisions: tuple[NearDuplicateDecision, ...]
    candidate_pair_count: int
    compared_pair_count: int


def normalize_for_comparison(text: str) -> str:
    """Normalize text for matching without using it as classifier content.

    Unicode compatibility normalization makes visually equivalent characters
    comparable. Case and whitespace differences are ignored. The source text is
    never modified by this function because strings are immutable and the
    normalized value is returned separately.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip().casefold()


def split_text_blocks(text: str) -> list[TextBlock]:
    """Split extracted page text at blank lines while preserving block order.

    Single line breaks remain inside a block because they can represent headings,
    list formatting, or wrapped prose. Empty blocks are discarded. Only outer
    whitespace is removed from each output block; the supplied page text itself
    is not changed.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Standardize line endings only in the temporary segmentation representation.
    segmentation_text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[TextBlock] = []
    for candidate in BLANK_LINE_RE.split(segmentation_text):
        block_text = candidate.strip()
        if not block_text:
            continue
        normalized = normalize_for_comparison(block_text)
        if not normalized:
            continue
        blocks.append(
            TextBlock(
                index=len(blocks),
                text=block_text,
                normalized=normalized,
                character_count=len(block_text),
                normalized_character_count=len(normalized),
            )
        )
    return blocks


def _is_edge_position(block_index: int, block_count: int) -> bool:
    """Return whether a block falls in the first or last page-edge window."""
    if block_count <= 0:
        return False
    edge_width = max(1, ceil(block_count * EDGE_PAGE_FRACTION))
    return block_index < edge_width or block_index >= block_count - edge_width


def recurring_block_statistics(
    pages: list[list[TextBlock]],
) -> dict[str, RecurringBlockStats]:
    """Measure block recurrence once per page and position across occurrences."""
    page_occurrences: Counter[str] = Counter()
    occurrence_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()

    for blocks in pages:
        seen_on_page: set[str] = set()
        for block in blocks:
            normalized = block.normalized
            occurrence_counts[normalized] += 1
            if _is_edge_position(block.index, len(blocks)):
                edge_counts[normalized] += 1
            seen_on_page.add(normalized)
        page_occurrences.update(seen_on_page)

    total_pages = len(pages)
    statistics: dict[str, RecurringBlockStats] = {}
    for normalized, occurrence_count in occurrence_counts.items():
        page_count = page_occurrences[normalized]
        edge_count = edge_counts[normalized]
        statistics[normalized] = RecurringBlockStats(
            normalized=normalized,
            page_count=page_count,
            page_fraction=page_count / total_pages if total_pages else 0.0,
            occurrence_count=occurrence_count,
            edge_occurrence_count=edge_count,
            edge_occurrence_fraction=edge_count / occurrence_count,
        )
    return statistics


def _recurring_candidates(
    pages: list[list[TextBlock]], statistics: dict[str, RecurringBlockStats]
) -> set[str]:
    """Return blocks meeting the absolute and proportional recurrence thresholds."""
    if not pages:
        return set()
    required_pages = max(MIN_RECURRING_PAGES, ceil(len(pages) * MIN_RECURRING_PAGE_FRACTION))
    return {
        normalized
        for normalized, stats in statistics.items()
        if stats.page_count >= required_pages
    }


def _interior_run_indexes(blocks: list[TextBlock], candidates: set[str]) -> set[int]:
    """Find consecutive runs of short recurring blocks, including interior menus."""
    removable: set[int] = set()
    current_run: list[int] = []

    def finish_run() -> None:
        if len(current_run) >= MIN_RECURRING_RUN_LENGTH:
            removable.update(current_run)
        current_run.clear()

    for block in blocks:
        qualifies = (
            block.normalized in candidates
            and block.character_count <= MAX_INTERIOR_BLOCK_CHARACTERS
        )
        if qualifies:
            current_run.append(block.index)
        else:
            finish_run()
    finish_run()
    return removable


def detect_recurring_boilerplate(
    pages: list[list[TextBlock]],
) -> list[list[BoilerplateDecision]]:
    """Identify conservative recurring boilerplate decisions for company pages.

    Recurrence alone is not sufficient. A recurring block is removable only when
    it occurs predominantly near page boundaries, or forms part of a run of at
    least three consecutive short recurring blocks. This protects isolated body
    headings and repeated substantive statements.
    """
    statistics = recurring_block_statistics(pages)
    candidates = _recurring_candidates(pages, statistics)
    decisions: list[list[BoilerplateDecision]] = []

    for page_index, blocks in enumerate(pages):
        run_indexes = _interior_run_indexes(blocks, candidates)
        page_decisions: list[BoilerplateDecision] = []
        for block in blocks:
            if block.normalized not in candidates:
                continue
            stats = statistics[block.normalized]
            reasons: list[str] = []
            if (
                block.character_count <= MAX_EDGE_BLOCK_CHARACTERS
                and stats.edge_occurrence_fraction >= MIN_EDGE_OCCURRENCE_FRACTION
                and _is_edge_position(block.index, len(blocks))
            ):
                reasons.append("recurring_at_page_edge")
            if block.index in run_indexes:
                reasons.append("recurring_block_run")
            if not reasons:
                continue
            page_decisions.append(
                BoilerplateDecision(
                    page_index=page_index,
                    block_index=block.index,
                    text=block.text,
                    normalized=block.normalized,
                    reasons=tuple(reasons),
                    recurring_page_count=stats.page_count,
                    recurring_page_fraction=stats.page_fraction,
                    edge_occurrence_fraction=stats.edge_occurrence_fraction,
                )
            )
        decisions.append(page_decisions)
    return decisions


def page_fingerprint(text: str) -> str:
    """Return a stable SHA-256 fingerprint of complete normalized page text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = normalize_for_comparison(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _valid_depth(record: dict[str, object]) -> int:
    depth = record.get("depth")
    if isinstance(depth, int) and not isinstance(depth, bool) and depth >= 0:
        return depth
    return 10**9


def _capture_rank(record: dict[str, object]) -> int:
    """Return a descending numeric timestamp rank, or zero when unavailable."""
    capture = record.get("capture_ts")
    if isinstance(capture, str) and capture.isdigit():
        return int(capture)
    if isinstance(capture, int) and not isinstance(capture, bool):
        return capture
    return 0


def _canonical_sort_key(page_index: int, record: dict[str, object]) -> tuple[object, ...]:
    text = record.get("text")
    source_text = text if isinstance(text, str) else ""
    substantive_characters = sum(not character.isspace() for character in source_text)
    url = record.get("url")
    stable_url = url if isinstance(url, str) else ""
    return (
        _valid_depth(record),
        -substantive_characters,
        -_capture_rank(record),
        stable_url,
        page_index,
    )


def detect_exact_duplicate_pages(records: list[dict[str, object]]) -> ExactDuplicateResult:
    """Group normalized-exact page texts and choose one canonical page per group.

    Empty texts are not grouped because Phase 2 should already have excluded them.
    A non-string text is rejected rather than silently converted, preserving the
    validation boundary between phases.
    """
    fingerprint_members: dict[str, list[int]] = defaultdict(list)
    for page_index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str):
            raise TypeError(f"record {page_index} text must be a string")
        if not normalize_for_comparison(text):
            continue
        fingerprint_members[page_fingerprint(text)].append(page_index)

    canonical_indexes: set[int] = set(range(len(records)))
    decisions: list[ExactDuplicateDecision] = []
    groups: list[ExactDuplicateGroup] = []

    for fingerprint, members in sorted(fingerprint_members.items()):
        if len(members) < 2:
            continue
        canonical = min(members, key=lambda index: _canonical_sort_key(index, records[index]))
        ordered_members = tuple(sorted(members))
        groups.append(
            ExactDuplicateGroup(
                fingerprint=fingerprint,
                canonical_page_index=canonical,
                member_page_indexes=ordered_members,
            )
        )
        for page_index in ordered_members:
            if page_index == canonical:
                continue
            canonical_indexes.discard(page_index)
            decisions.append(
                ExactDuplicateDecision(
                    page_index=page_index,
                    canonical_page_index=canonical,
                    fingerprint=fingerprint,
                    group_size=len(ordered_members),
                )
            )

    return ExactDuplicateResult(
        canonical_page_indexes=tuple(sorted(canonical_indexes)),
        decisions=tuple(sorted(decisions, key=lambda decision: decision.page_index)),
        groups=tuple(sorted(groups, key=lambda group: group.member_page_indexes)),
    )


def word_shingles(text: str, size: int = NEAR_DUPLICATE_SHINGLE_SIZE) -> frozenset[str]:
    """Return normalized, ordered word shingles for near-duplicate comparison."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if size < 1:
        raise ValueError("shingle size must be at least 1")
    words = WORD_RE.findall(normalize_for_comparison(text))
    if len(words) < size:
        return frozenset()
    return frozenset(" ".join(words[index : index + size]) for index in range(len(words) - size + 1))


def _comparison_texts_without_boilerplate(
    records: list[dict[str, object]],
) -> list[str]:
    """Build comparison-only page texts after provisional boilerplate omission."""
    pages: list[list[TextBlock]] = []
    for page_index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str):
            raise TypeError(f"record {page_index} text must be a string")
        pages.append(split_text_blocks(text))

    boilerplate = detect_recurring_boilerplate(pages)
    comparison_texts: list[str] = []
    for blocks, decisions in zip(pages, boilerplate):
        removed_indexes = {decision.block_index for decision in decisions}
        comparison_texts.append(
            "\n\n".join(
                block.normalized for block in blocks if block.index not in removed_indexes
            )
        )
    return comparison_texts


def _near_duplicate_candidates(shingle_sets: list[frozenset[str]]) -> set[tuple[int, int]]:
    """Generate likely pairs using stable, rare shingles as page anchors."""
    postings: dict[str, list[int]] = defaultdict(list)
    for page_index, shingles in enumerate(shingle_sets):
        for shingle in shingles:
            postings[shingle].append(page_index)

    page_count = len(shingle_sets)
    max_anchor_pages = max(2, ceil(page_count * NEAR_DUPLICATE_MAX_ANCHOR_PAGE_FRACTION))
    selected_anchors: list[set[str]] = []
    for shingles in shingle_sets:
        eligible = [
            shingle for shingle in shingles if 1 < len(postings[shingle]) <= max_anchor_pages
        ]
        eligible.sort(key=lambda shingle: (len(postings[shingle]), shingle))
        selected_anchors.append(set(eligible[:NEAR_DUPLICATE_ANCHORS_PER_PAGE]))

    candidates: set[tuple[int, int]] = set()
    for page_index, anchors in enumerate(selected_anchors):
        for anchor in anchors:
            for other_index in postings[anchor]:
                if other_index == page_index:
                    continue
                candidates.add(tuple(sorted((page_index, other_index))))
    return candidates


def _jaccard(left: frozenset[str], right: frozenset[str]) -> tuple[float, int]:
    shared = len(left.intersection(right))
    union = len(left) + len(right) - shared
    return (shared / union if union else 0.0), shared


def detect_near_duplicate_pages(records: list[dict[str, object]]) -> NearDuplicateResult:
    """Detect high-similarity pages after ignoring recurring boilerplate.

    This function should receive exact-canonical records. Rare shingles only
    generate candidate pairs; every final decision is based on exact Jaccard
    similarity over each page's complete comparison shingle set.
    """
    comparison_texts = _comparison_texts_without_boilerplate(records)
    word_counts = [len(WORD_RE.findall(text)) for text in comparison_texts]
    shingle_sets = [word_shingles(text) for text in comparison_texts]
    eligible = {
        index
        for index, shingles in enumerate(shingle_sets)
        if word_counts[index] >= NEAR_DUPLICATE_MIN_WORDS
        and len(shingles) >= NEAR_DUPLICATE_MIN_SHINGLES
    }
    candidates = {
        pair
        for pair in _near_duplicate_candidates(shingle_sets)
        if pair[0] in eligible and pair[1] in eligible
    }

    similar_pairs: dict[tuple[int, int], tuple[float, float, int]] = {}
    compared_pair_count = 0
    for left, right in sorted(candidates):
        left_size = len(shingle_sets[left])
        right_size = len(shingle_sets[right])
        length_ratio = min(left_size, right_size) / max(left_size, right_size)
        if length_ratio < NEAR_DUPLICATE_MIN_LENGTH_RATIO:
            continue
        compared_pair_count += 1
        similarity, shared = _jaccard(shingle_sets[left], shingle_sets[right])
        if similarity >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
            similar_pairs[(left, right)] = (similarity, length_ratio, shared)

    preferred_order = sorted(
        range(len(records)), key=lambda index: _canonical_sort_key(index, records[index])
    )
    canonical_indexes: set[int] = set()
    decisions: list[NearDuplicateDecision] = []
    for page_index in preferred_order:
        matching_canonicals: list[tuple[float, int, float, int]] = []
        for canonical in canonical_indexes:
            pair = tuple(sorted((page_index, canonical)))
            evidence = similar_pairs.get(pair)
            if evidence is None:
                continue
            similarity, length_ratio, shared = evidence
            matching_canonicals.append((similarity, canonical, length_ratio, shared))
        if not matching_canonicals:
            canonical_indexes.add(page_index)
            continue
        similarity, canonical, length_ratio, shared = max(
            matching_canonicals,
            key=lambda item: (item[0], -preferred_order.index(item[1])),
        )
        decisions.append(
            NearDuplicateDecision(
                page_index=page_index,
                canonical_page_index=canonical,
                jaccard_similarity=similarity,
                length_ratio=length_ratio,
                shared_shingles=shared,
                page_shingles=len(shingle_sets[page_index]),
                canonical_shingles=len(shingle_sets[canonical]),
            )
        )

    return NearDuplicateResult(
        canonical_page_indexes=tuple(sorted(canonical_indexes)),
        decisions=tuple(sorted(decisions, key=lambda decision: decision.page_index)),
        candidate_pair_count=len(candidates),
        compared_pair_count=compared_pair_count,
    )


def _write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _page_identity(record: dict[str, object], page_index: int) -> dict[str, object]:
    return {
        "page_index": page_index,
        "company": record.get("company"),
        "slug": record.get("slug"),
        "url": record.get("url"),
        "title": record.get("title"),
        "depth": record.get("depth"),
        "capture_ts": record.get("capture_ts"),
    }


def process_phase3_company(source_path: Path, output_dir: Path) -> dict[str, Any]:
    """Apply Phase 3 decisions to one Phase 2 retained company file."""
    slug = source_path.stem
    records: list[dict[str, object]] = []
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise TypeError(f"{source_path}:{line_number} must contain a JSON object")
            if not isinstance(parsed.get("text"), str):
                raise TypeError(f"{source_path}:{line_number} text must be a string")
            records.append(parsed)

    exact = detect_exact_duplicate_pages(records)
    exact_canonical_original_indexes = list(exact.canonical_page_indexes)
    exact_canonical_records = [records[index] for index in exact_canonical_original_indexes]

    canonical_blocks = [
        split_text_blocks(record["text"])  # type: ignore[arg-type]
        for record in exact_canonical_records
    ]
    boilerplate_decisions = detect_recurring_boilerplate(canonical_blocks)
    near = detect_near_duplicate_pages(exact_canonical_records)
    final_local_indexes = set(near.canonical_page_indexes)

    cleaned_path = output_dir / "cleaned" / source_path.name
    duplicate_path = output_dir / "duplicates" / source_path.name
    boilerplate_path = output_dir / "boilerplate" / source_path.name
    profile_path = output_dir / "profiles" / f"{slug}.json"
    for path in (cleaned_path, duplicate_path, boilerplate_path, profile_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    source_characters = sum(len(record["text"]) for record in records)  # type: ignore[arg-type]
    cleaned_characters = 0
    removed_boilerplate_characters = 0
    removed_boilerplate_blocks = 0
    empty_after_boilerplate = 0
    written_pages = 0

    exact_by_page = {decision.page_index: decision for decision in exact.decisions}
    near_by_local_page = {decision.page_index: decision for decision in near.decisions}
    boilerplate_groups: dict[tuple[object, ...], dict[str, Any]] = {}

    with (
        cleaned_path.open("w", encoding="utf-8") as cleaned_output,
        duplicate_path.open("w", encoding="utf-8") as duplicate_output,
        boilerplate_path.open("w", encoding="utf-8") as boilerplate_output,
    ):
        for original_index, decision in sorted(exact_by_page.items()):
            canonical_record = records[decision.canonical_page_index]
            audit = {
                **_page_identity(records[original_index], original_index),
                "status": "excluded",
                "primary_reason": decision.reason,
                "canonical": _page_identity(canonical_record, decision.canonical_page_index),
                "fingerprint": decision.fingerprint,
                "group_size": decision.group_size,
            }
            _write_jsonl_row(duplicate_output, audit)

        for local_index, decision in sorted(near_by_local_page.items()):
            original_index = exact_canonical_original_indexes[local_index]
            canonical_original_index = exact_canonical_original_indexes[
                decision.canonical_page_index
            ]
            audit = {
                **_page_identity(records[original_index], original_index),
                "status": "excluded",
                "primary_reason": decision.reason,
                "canonical": _page_identity(
                    records[canonical_original_index], canonical_original_index
                ),
                "similarity": {
                    "jaccard": decision.jaccard_similarity,
                    "length_ratio": decision.length_ratio,
                    "shared_shingles": decision.shared_shingles,
                    "page_shingles": decision.page_shingles,
                    "canonical_shingles": decision.canonical_shingles,
                },
            }
            _write_jsonl_row(duplicate_output, audit)

        for local_index, (record, blocks, decisions) in enumerate(
            zip(exact_canonical_records, canonical_blocks, boilerplate_decisions)
        ):
            if local_index not in final_local_indexes:
                continue
            original_index = exact_canonical_original_indexes[local_index]
            removed_indexes = {decision.block_index for decision in decisions}
            kept_blocks = [block for block in blocks if block.index not in removed_indexes]
            cleaned_text = "\n\n".join(block.text for block in kept_blocks)

            for decision in decisions:
                removed_boilerplate_blocks += 1
                removed_boilerplate_characters += len(decision.text)
                group_key = (
                    decision.normalized,
                    decision.reasons,
                    decision.recurring_page_count,
                    decision.recurring_page_fraction,
                    decision.edge_occurrence_fraction,
                )
                audit = boilerplate_groups.setdefault(group_key, {
                    "company": record.get("company"),
                    "slug": record.get("slug"),
                    "status": "removed_block",
                    "primary_reason": "recurring_boilerplate",
                    "sample_text": decision.text,
                    "normalized": decision.normalized,
                    "reasons": list(decision.reasons),
                    "recurring_page_count": decision.recurring_page_count,
                    "recurring_page_fraction": decision.recurring_page_fraction,
                    "edge_occurrence_fraction": decision.edge_occurrence_fraction,
                    "occurrences": [],
                })
                audit["occurrences"].append([original_index, decision.block_index])

            if not normalize_for_comparison(cleaned_text):
                empty_after_boilerplate += 1
                _write_jsonl_row(
                    duplicate_output,
                    {
                        **_page_identity(record, original_index),
                        "status": "excluded",
                        "primary_reason": "empty_after_boilerplate",
                        "removed_block_count": len(decisions),
                    },
                )
                continue

            output_record = dict(record)
            output_record["text"] = cleaned_text
            output_record["n_chars"] = len(cleaned_text)
            output_record["_phase3"] = {
                "status": "cleaned",
                "source_page_index": original_index,
                "original_text_characters": len(record["text"]),  # type: ignore[arg-type]
                "cleaned_text_characters": len(cleaned_text),
                "original_block_count": len(blocks),
                "removed_boilerplate_block_count": len(decisions),
                "retained_block_count": len(kept_blocks),
                "exact_duplicate_canonical": True,
                "near_duplicate_canonical": True,
            }
            _write_jsonl_row(cleaned_output, output_record)
            written_pages += 1
            cleaned_characters += len(cleaned_text)

        for group_key in sorted(boilerplate_groups, key=lambda key: (str(key[0]), str(key[1]))):
            audit = boilerplate_groups[group_key]
            audit["occurrence_count"] = len(audit["occurrences"])
            _write_jsonl_row(boilerplate_output, audit)

    profile: dict[str, Any] = {
        "slug": slug,
        "source_file": str(source_path),
        "output_files": {
            "cleaned": str(cleaned_path.relative_to(output_dir)),
            "duplicates": str(duplicate_path.relative_to(output_dir)),
            "boilerplate": str(boilerplate_path.relative_to(output_dir)),
        },
        "counts": {
            "phase2_pages": len(records),
            "exact_duplicate_pages": len(exact.decisions),
            "exact_duplicate_groups": len(exact.groups),
            "exact_canonical_pages": len(exact.canonical_page_indexes),
            "near_duplicate_pages": len(near.decisions),
            "empty_after_boilerplate_pages": empty_after_boilerplate,
            "phase3_cleaned_pages": written_pages,
            "removed_boilerplate_blocks": removed_boilerplate_blocks,
            "near_duplicate_candidate_pairs": near.candidate_pair_count,
            "near_duplicate_compared_pairs": near.compared_pair_count,
        },
        "character_totals": {
            "phase2_source_text": source_characters,
            "phase3_cleaned_text": cleaned_characters,
            "removed_boilerplate_block_text": removed_boilerplate_characters,
        },
        "configuration": {
            "recurring_minimum_pages": MIN_RECURRING_PAGES,
            "recurring_minimum_page_fraction": MIN_RECURRING_PAGE_FRACTION,
            "edge_page_fraction": EDGE_PAGE_FRACTION,
            "edge_occurrence_fraction": MIN_EDGE_OCCURRENCE_FRACTION,
            "interior_run_minimum_blocks": MIN_RECURRING_RUN_LENGTH,
            "near_duplicate_shingle_size": NEAR_DUPLICATE_SHINGLE_SIZE,
            "near_duplicate_minimum_words": NEAR_DUPLICATE_MIN_WORDS,
            "near_duplicate_minimum_length_ratio": NEAR_DUPLICATE_MIN_LENGTH_RATIO,
            "near_duplicate_jaccard_threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
        },
    }
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile


def process_phase3_all(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Apply Phase 3 to all Phase 2 retained company JSONL files."""
    source_paths = sorted(input_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not source_paths:
        raise FileNotFoundError(f"No retained company JSONL files found in {input_dir}")
    profiles = [process_phase3_company(path, output_dir) for path in source_paths]

    count_totals: Counter[str] = Counter()
    character_totals: Counter[str] = Counter()
    for profile in profiles:
        count_totals.update(profile["counts"])
        character_totals.update(profile["character_totals"])
    summary = {
        "phase": 3,
        "company_count": len(profiles),
        "companies": [profile["slug"] for profile in profiles],
        "counts": dict(sorted(count_totals.items())),
        "character_totals": dict(sorted(character_totals.items())),
        "configuration": profiles[0]["configuration"],
        "scope": {
            "depth_filtering": False,
            "language_filtering": False,
            "token_counting": False,
            "page_ranking": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
