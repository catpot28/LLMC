"""Phase 5 deterministic selection under a per-company text-token budget.

Selection is target-neutral. Crawl depth is authoritative, page type orders
pages within a depth, and shorter whole pages precede longer pages within a
type. Depth-3 pages are considered only after the depth 0-2 primary pool.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .clean_deduplicate import split_text_blocks
from .select_rank_pages import (
    DEFAULT_BUDGET_ENCODING,
    MAXIMUM_PAGE_CONTRIBUTION_TOKENS,
    _budget_encoding,
    count_budget_tokens,
)


ROOT = Path(__file__).resolve().parent.parent
TARGET_CONTENT_TOKENS = 45_000

CATEGORY_PRIORITY = (
    "homepage",
    "about_company",
    "products_services",
    "industries_markets",
    "cases_projects",
    "research_reports",
    "technical_documentation",
    "careers",
    "contact_locations",
    "news_articles",
    "other",
)
CATEGORY_RANK = {category: rank for rank, category in enumerate(CATEGORY_PRIORITY)}


def _page_type(record: dict[str, Any]) -> str:
    return (
        record.get("_phase4", {})
        .get("page_type", {})
        .get("page_type", "other")
    )


def _page_tokens(record: dict[str, Any]) -> int:
    value = (
        record.get("_phase4", {})
        .get("budget_candidate", {})
        .get("page_token_count")
    )
    if not isinstance(value, int):
        raise TypeError("budget candidate metadata requires integer page_token_count")
    return value


def candidate_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return the documented stable priority for one candidate page."""
    depth = record.get("depth")
    if not isinstance(depth, int) or isinstance(depth, bool):
        depth = 10**9
    page_type = _page_type(record)
    tokens = _page_tokens(record)
    return (
        depth,
        CATEGORY_RANK.get(page_type, len(CATEGORY_PRIORITY)),
        min(tokens, MAXIMUM_PAGE_CONTRIBUTION_TOKENS),
        str(record.get("url", "")),
    )


def _token_prefix(text: str, limit: int, encoding_name: str) -> str:
    """Return at most ``limit`` tokens, preferring a complete text block."""
    if limit <= 0:
        return ""
    encoding = _budget_encoding(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) <= limit:
        return text
    return encoding.decode(tokens[:limit]).rstrip()


def oversized_contribution(
    text: str,
    limit: int,
    encoding_name: str = DEFAULT_BUDGET_ENCODING,
) -> tuple[str, list[int], str]:
    """Build a bounded contribution in source order from cleaned text blocks."""
    if limit <= 0:
        return "", [], "none"
    selected: list[str] = []
    selected_indexes: list[int] = []
    used = 0
    boundary = "block_boundary"
    for block in split_text_blocks(text):
        separator = "\n\n" if selected else ""
        candidate = separator + block.text
        candidate_tokens = count_budget_tokens(candidate, encoding_name)
        if used + candidate_tokens <= limit:
            selected.append(block.text)
            selected_indexes.append(block.index)
            used += candidate_tokens
            continue
        remaining = limit - used
        prefix = _token_prefix(candidate, remaining, encoding_name)
        if prefix:
            if selected and prefix.startswith("\n\n"):
                prefix = prefix[2:]
            selected.append(prefix)
            selected_indexes.append(block.index)
            boundary = "token_limit"
        break
    return "\n\n".join(selected), selected_indexes, boundary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise TypeError(f"{path}:{line_number} requires a page object with text")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def select_company(
    primary_path: Path,
    reserve_path: Path,
    output_dir: Path,
    target_tokens: int = TARGET_CONTENT_TOKENS,
    page_cap: int = MAXIMUM_PAGE_CONTRIBUTION_TOKENS,
    encoding_name: str = DEFAULT_BUDGET_ENCODING,
) -> dict[str, Any]:
    """Select one company deterministically, considering reserve only second."""
    if target_tokens <= 0 or page_cap <= 0:
        raise ValueError("target_tokens and page_cap must be positive")
    slug = primary_path.stem
    primary = sorted(_load_jsonl(primary_path), key=candidate_sort_key)
    reserve = sorted(_load_jsonl(reserve_path), key=candidate_sort_key)
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    assembled_parts: list[str] = []

    for pool, candidates in (("primary_depth_0_2", primary), ("depth_3_reserve", reserve)):
        for order, record in enumerate(candidates):
            current_text = "\n\n".join(assembled_parts)
            current_tokens = count_budget_tokens(current_text, encoding_name)
            remaining = target_tokens - current_tokens
            if remaining <= 0:
                audit.append({
                    "status": "not_selected",
                    "primary_reason": "company_token_budget_reached",
                    "pool": pool,
                    "candidate_order": order,
                    "url": record.get("url"),
                    "depth": record.get("depth"),
                    "page_type": _page_type(record),
                    "page_token_count": _page_tokens(record),
                })
                continue

            page_tokens = _page_tokens(record)
            oversized = page_tokens > page_cap
            if oversized:
                contribution_limit = min(page_cap, remaining)
                contribution, block_indexes, boundary = oversized_contribution(
                    record["text"], contribution_limit, encoding_name
                )
                selection_mode = "selected_blocks"
            else:
                contribution = record["text"]
                block_indexes = []
                boundary = "whole_page"
                selection_mode = "whole_page"

            combined = "\n\n".join(assembled_parts + [contribution]) if contribution else current_text
            combined_tokens = count_budget_tokens(combined, encoding_name)
            if not contribution or combined_tokens > target_tokens:
                audit.append({
                    "status": "not_selected",
                    "primary_reason": "whole_page_does_not_fit" if not oversized else "no_oversized_contribution_fits",
                    "pool": pool,
                    "candidate_order": order,
                    "url": record.get("url"),
                    "depth": record.get("depth"),
                    "page_type": _page_type(record),
                    "page_token_count": page_tokens,
                })
                continue

            previous_tokens = current_tokens
            contributed_tokens = combined_tokens - previous_tokens
            output_record = dict(record)
            output_record["text"] = contribution
            phase5 = {
                "status": "selected",
                "pool": pool,
                "candidate_order": order,
                "page_type": _page_type(record),
                "original_page_token_count": page_tokens,
                "selected_token_contribution": contributed_tokens,
                "oversized": oversized,
                "selection_mode": selection_mode,
                "selected_block_indexes": block_indexes,
                "ending_boundary": boundary,
            }
            output_record["_phase5"] = phase5
            selected.append(output_record)
            assembled_parts.append(contribution)
            if oversized and contributed_tokens < page_tokens:
                audit.append({
                    "status": "partially_selected",
                    "primary_reason": "per_page_token_cap",
                    "pool": pool,
                    "candidate_order": order,
                    "url": record.get("url"),
                    "depth": record.get("depth"),
                    "page_type": _page_type(record),
                    "page_token_count": page_tokens,
                    "selected_token_contribution": contributed_tokens,
                    "maximum_page_contribution_tokens": page_cap,
                })

    final_text = "\n\n".join(assembled_parts)
    final_tokens = count_budget_tokens(final_text, encoding_name)
    selected_path = output_dir / "selected_pages" / f"{slug}.jsonl"
    audit_path = output_dir / "selection_audit" / f"{slug}.jsonl"
    final_path = output_dir / "final" / f"{slug}.json"
    profile_path = output_dir / "profiles" / f"{slug}.json"
    _write_jsonl(selected_path, selected)
    _write_jsonl(audit_path, audit)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    company = selected[0].get("company") if selected else None
    final_path.write_text(json.dumps({
        "company": company,
        "slug": slug,
        "tokenizer": encoding_name,
        "content_token_count": final_tokens,
        "target_content_tokens": target_tokens,
        "text": final_text,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pools = Counter(row["_phase5"]["pool"] for row in selected)
    types = Counter(row["_phase5"]["page_type"] for row in selected)
    profile = {
        "slug": slug,
        "tokenizer": encoding_name,
        "target_content_tokens": target_tokens,
        "maximum_page_contribution_tokens": page_cap,
        "content_token_count": final_tokens,
        "unused_token_capacity": target_tokens - final_tokens,
        "selected_pages": len(selected),
        "partially_selected_pages": sum(row["status"] == "partially_selected" for row in audit),
        "selected_pools": dict(sorted(pools.items())),
        "selected_page_types": dict(sorted(types.items())),
        "output_files": {
            "final": str(final_path.relative_to(output_dir)),
            "selected_pages": str(selected_path.relative_to(output_dir)),
            "selection_audit": str(audit_path.relative_to(output_dir)),
        },
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def select_all(
    primary_dir: Path,
    reserve_dir: Path,
    output_dir: Path,
    target_tokens: int = TARGET_CONTENT_TOKENS,
    page_cap: int = MAXIMUM_PAGE_CONTRIBUTION_TOKENS,
    encoding_name: str = DEFAULT_BUDGET_ENCODING,
) -> dict[str, Any]:
    primary_paths = sorted(primary_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not primary_paths:
        raise FileNotFoundError(f"No budget primary files found in {primary_dir}")
    profiles = []
    for primary_path in primary_paths:
        reserve_path = reserve_dir / primary_path.name
        if not reserve_path.exists():
            raise FileNotFoundError(f"Missing reserve file: {reserve_path}")
        profiles.append(select_company(primary_path, reserve_path, output_dir, target_tokens, page_cap, encoding_name))
    tokens = [profile["content_token_count"] for profile in profiles]
    summary = {
        "phase": "5-deterministic-selection",
        "company_count": len(profiles),
        "tokenizer": encoding_name,
        "target_content_tokens": target_tokens,
        "maximum_page_contribution_tokens": page_cap,
        "category_priority": list(CATEGORY_PRIORITY),
        "ordering": ["crawl_depth", "page_type", "shorter_effective_page", "url"],
        "depth3_considered_after_primary": True,
        "companies_at_target": sum(value == target_tokens for value in tokens),
        "companies_below_target": sum(value < target_tokens for value in tokens),
        "content_tokens": {"minimum": min(tokens), "maximum": max(tokens), "total": sum(tokens)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
