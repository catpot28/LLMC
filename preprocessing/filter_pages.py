"""Validate scraped pages and remove deterministic non-content pages.

This module implements preprocessing phases 1 and 2 only. It deliberately does
not clean page text, deduplicate pages, detect language, count tokens, or select
pages for a token budget.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data" / "scraped_data" / "clean"
DEFAULT_SELECTED_CSV = ROOT / "data" / "scraped_data" / "enum" / "selected.csv"

# This threshold only identifies pages that cannot carry meaningful prose. It is
# not text cleaning and does not change retained text.
MIN_SUBSTANTIVE_CHARS = 200

LEGAL_URL_RE = re.compile(
    r"(?:^|[-_/])(?:"
    r"privacy(?:[-_](?:policy|notice|statement))?|data[-_]protection|gdpr|"
    r"terms(?:[-_](?:and[-_](?:conditions|use)|conditions|of[-_]use|service))?|"
    r"conditions[-_]of[-_]use|cookie(?:s|[-_](?:policy|settings|preferences))?|"
    r"legal(?:[-_](?:notice|disclaimer|information|policy))?|disclaimer|"
    r"imprint|impressum|accessibility(?:[-_](?:policy|statement))?|"
    r"compliance[-_]policy|regulatory[-_]policy|policies"
    r")(?:$|[-_/.?])",
    re.IGNORECASE,
)
LEGAL_TITLE_RE = re.compile(
    r"^(?:privacy (?:policy|notice|statement)|data protection(?: policy)?|"
    r"terms (?:and conditions|of use|of service)|cookie (?:policy|settings|preferences)|"
    r"legal (?:notice|disclaimer|information|policy)|disclaimer|imprint|impressum|"
    r"accessibility (?:policy|statement)|compliance policy|regulatory policy)"
    r"(?:\s*[|\-–—:]|$)",
    re.IGNORECASE,
)

LOGIN_URL_RE = re.compile(
    r"(?:^|[-_/])(?:login|log[-_]?in|signin|sign[-_]?in|signup|sign[-_]?up|"
    r"register|registration|account|my[-_]?account|password|forgot[-_]?password|"
    r"reset[-_]?password|auth)(?:$|[-_/.?])",
    re.IGNORECASE,
)
LOGIN_TITLE_RE = re.compile(
    r"^(?:log ?in|sign ?in|sign ?up|register|registration|my account|"
    r"forgot password|reset password)(?:\s*[|\-–—:]|$)",
    re.IGNORECASE,
)

UTILITY_URL_RE = re.compile(
    r"(?:^|[-_/])(?:search|search-results?|sitemap|tags?|archives?|pagination|"
    r"page/\d+|feed|rss|wp-json|print|share|unsubscribe|preferences|"
    r"compare|cart|checkout)(?:$|[-_/.?])",
    re.IGNORECASE,
)
UTILITY_TITLE_RE = re.compile(
    r"^(?:search(?: results?)?|sitemap|tags?|archives?|page not found|"
    r"unsubscribe|shopping cart|checkout)(?:\s*[|\-–—:]|$)",
    re.IGNORECASE,
)

ERROR_TITLE_RE = re.compile(
    r"^(?:error|access denied|forbidden|unauthorized|page not found|not found|"
    r"404|403|500|service unavailable|temporarily unavailable|just a moment)"
    r"(?:\s*[|\-–—:]|$)",
    re.IGNORECASE,
)
ERROR_TEXT_RE = re.compile(
    r"\b(?:access denied|you do not have permission|page (?:could not be found|does not exist)|"
    r"the requested page (?:was not found|is unavailable)|service unavailable)\b",
    re.IGNORECASE,
)

INTERFACE_PHRASE_RE = re.compile(
    r"\b(?:enable javascript|accept all cookies|manage cookie preferences|"
    r"enter your (?:email|password)|forgot your password|create an account|"
    r"please sign in|verification code|required field)\b",
    re.IGNORECASE,
)


def compact_identity(value: Any) -> str:
    """Return a normalized string for matching, without changing source text."""
    return " ".join(str(value or "").split()).strip()


def url_identity(url: str) -> str:
    """Return decoded URL components suitable for deterministic rule matching."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return unquote(url).lower()
    return unquote(f"{parsed.path}?{parsed.query}").lower()


def substantive_char_count(text: str) -> int:
    """Count non-whitespace characters without altering the supplied text."""
    return sum(not char.isspace() for char in text)


def load_selected_metadata(path: Path) -> dict[tuple[str, str], set[int]]:
    """Load allowed crawl depths keyed by (slug, URL)."""
    metadata: dict[tuple[str, str], set[int]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            slug = compact_identity(row.get("slug"))
            url = compact_identity(row.get("url"))
            try:
                depth = int(row.get("depth", ""))
            except (TypeError, ValueError):
                continue
            if slug and url:
                metadata[(slug, url)].add(depth)
    return dict(metadata)


def validate_record(
    record: dict[str, Any], expected_slug: str, selected: dict[tuple[str, str], set[int]]
) -> tuple[list[str], str]:
    """Validate required fields and return issues plus metadata-match status."""
    issues: list[str] = []
    required = ("company", "slug", "url", "depth", "mime", "text")
    for field in required:
        if field not in record or record[field] is None or record[field] == "":
            issues.append(f"missing_{field}")

    slug = record.get("slug")
    if slug not in (None, "") and not isinstance(slug, str):
        issues.append("invalid_slug_type")
    elif slug and slug != expected_slug:
        issues.append("slug_filename_mismatch")

    company = record.get("company")
    if company not in (None, "") and not isinstance(company, str):
        issues.append("invalid_company_type")

    url = record.get("url")
    if url not in (None, ""):
        if not isinstance(url, str):
            issues.append("invalid_url_type")
        else:
            try:
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    issues.append("invalid_url")
            except ValueError:
                issues.append("invalid_url")

    depth = record.get("depth")
    if depth not in (None, "") and (not isinstance(depth, int) or isinstance(depth, bool) or depth < 0):
        issues.append("invalid_depth")

    mime = record.get("mime")
    if mime not in (None, "") and not isinstance(mime, str):
        issues.append("invalid_mime_type")

    text = record.get("text")
    if text is not None and not isinstance(text, str):
        issues.append("invalid_text_type")

    if not isinstance(slug, str) or not isinstance(url, str):
        return issues, "not_checkable"
    selected_depths = selected.get((slug, url))
    if selected_depths is None:
        issues.append("metadata_url_missing")
        return issues, "url_missing"
    if not isinstance(depth, int) or isinstance(depth, bool):
        return issues, "not_checkable"
    if depth not in selected_depths:
        issues.append("metadata_depth_mismatch")
        return issues, "depth_mismatch"
    return issues, "matched"


def classify_page(record: dict[str, Any], issues: Iterable[str]) -> tuple[str, list[str]]:
    """Return the primary exclusion reason and all matched filtering rules."""
    issue_set = set(issues)
    fatal_validation = {
        "missing_company",
        "missing_slug",
        "missing_url",
        "missing_depth",
        "missing_mime",
        "invalid_slug_type",
        "invalid_company_type",
        "invalid_url_type",
        "invalid_url",
        "invalid_depth",
        "invalid_mime_type",
        "invalid_text_type",
    }
    if issue_set & fatal_validation:
        return "malformed_record", ["validation_failed"]

    mime = compact_identity(record.get("mime")).lower().split(";", 1)[0]
    text_value = record.get("text")
    text = text_value if isinstance(text_value, str) else ""
    title = compact_identity(record.get("title"))
    url = compact_identity(record.get("url"))
    identity = url_identity(url)
    matches: list[str] = []

    note = compact_identity(record.get("note")).lower()
    if mime in {"pdf", "application/pdf"} or "pdf" in note:
        matches.append("pdf_placeholder")
    if mime not in {"html", "text/html", "application/xhtml+xml", "pdf", "application/pdf"}:
        matches.append("unsupported_mime")
    if LEGAL_URL_RE.search(identity) or LEGAL_TITLE_RE.search(title):
        matches.append("legal_or_policy")
    if LOGIN_URL_RE.search(identity) or LOGIN_TITLE_RE.search(title):
        matches.append("login_or_account")
    if UTILITY_URL_RE.search(identity) or UTILITY_TITLE_RE.search(title):
        matches.append("utility_page")
    if ERROR_TITLE_RE.search(title) or (len(text) <= 2000 and ERROR_TEXT_RE.search(text)):
        matches.append("error_or_access_denied")
    if not text.strip():
        matches.append("empty_text")
    elif substantive_char_count(text) < MIN_SUBSTANTIVE_CHARS:
        matches.append("negligible_text")
    elif len(text) <= 1500 and INTERFACE_PHRASE_RE.search(text):
        matches.append("interface_only")

    precedence = (
        "pdf_placeholder",
        "unsupported_mime",
        "legal_or_policy",
        "login_or_account",
        "utility_page",
        "error_or_access_denied",
        "empty_text",
        "negligible_text",
        "interface_only",
    )
    primary = next((reason for reason in precedence if reason in matches), "retained")
    return primary, matches


def audit_record(
    *,
    line_number: int,
    record: dict[str, Any] | None,
    status: str,
    reason: str,
    rules: list[str],
    validation_issues: list[str],
    metadata_match: str,
) -> dict[str, Any]:
    """Create a compact, provenance-preserving page decision."""
    source = record or {}
    text = source.get("text")
    return {
        "line_number": line_number,
        "status": status,
        "primary_reason": reason,
        "matched_rules": rules,
        "validation_issues": validation_issues,
        "metadata_match": metadata_match,
        "company": source.get("company"),
        "slug": source.get("slug"),
        "url": source.get("url"),
        "title": source.get("title"),
        "depth": source.get("depth"),
        "mime": source.get("mime"),
        "text_chars": len(text) if isinstance(text, str) else 0,
    }


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def process_company(
    source_path: Path,
    selected: dict[tuple[str, str], set[int]],
    output_dir: Path,
) -> dict[str, Any]:
    """Process one company file and return its profile."""
    slug = source_path.stem
    retained_path = output_dir / "retained" / source_path.name
    excluded_path = output_dir / "excluded" / source_path.name
    profile_path = output_dir / "profiles" / f"{slug}.json"
    for path in (retained_path, excluded_path, profile_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    depths: Counter[str] = Counter()
    mimes: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    metadata_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    source_chars = 0
    retained_chars = 0

    with (
        source_path.open(encoding="utf-8", errors="replace") as source,
        retained_path.open("w", encoding="utf-8") as retained,
        excluded_path.open("w", encoding="utf-8") as excluded,
    ):
        for line_number, line in enumerate(source, start=1):
            counts["total_records"] += 1
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                counts["malformed_records"] += 1
                exclusion_counts["invalid_json"] += 1
                write_jsonl_row(
                    excluded,
                    audit_record(
                        line_number=line_number,
                        record=None,
                        status="excluded",
                        reason="invalid_json",
                        rules=["json_parse_failed"],
                        validation_issues=[f"invalid_json:{exc.msg}"],
                        metadata_match="not_checkable",
                    ),
                )
                continue
            if not isinstance(parsed, dict):
                counts["malformed_records"] += 1
                exclusion_counts["malformed_record"] += 1
                write_jsonl_row(
                    excluded,
                    audit_record(
                        line_number=line_number,
                        record=None,
                        status="excluded",
                        reason="malformed_record",
                        rules=["record_not_object"],
                        validation_issues=["record_not_object"],
                        metadata_match="not_checkable",
                    ),
                )
                continue

            text = parsed.get("text")
            if isinstance(text, str):
                source_chars += len(text)
            depth = parsed.get("depth")
            depths[str(depth) if depth is not None else "missing"] += 1
            mimes[compact_identity(parsed.get("mime")).lower() or "missing"] += 1

            issues, metadata_match = validate_record(parsed, slug, selected)
            for issue in issues:
                validation_counts[issue] += 1
                if issue.startswith("missing_"):
                    missing_fields[issue.removeprefix("missing_")] += 1
            metadata_counts[metadata_match] += 1

            reason, rules = classify_page(parsed, issues)
            if reason == "retained":
                counts["retained_records"] += 1
                retained_chars += len(text) if isinstance(text, str) else 0
                output_record = dict(parsed)
                output_record["_preprocessing"] = {
                    "status": "retained",
                    "line_number": line_number,
                    "validation_issues": issues,
                    "metadata_match": metadata_match,
                }
                write_jsonl_row(retained, output_record)
            else:
                counts["excluded_records"] += 1
                if reason == "malformed_record":
                    counts["malformed_records"] += 1
                if reason == "pdf_placeholder":
                    counts["pdf_records"] += 1
                exclusion_counts[reason] += 1
                write_jsonl_row(
                    excluded,
                    audit_record(
                        line_number=line_number,
                        record=parsed,
                        status="excluded",
                        reason=reason,
                        rules=rules,
                        validation_issues=issues,
                        metadata_match=metadata_match,
                    ),
                )

    profile: dict[str, Any] = {
        "slug": slug,
        "source_file": str(source_path),
        "output_files": {
            "retained": str(retained_path.relative_to(output_dir)),
            "excluded": str(excluded_path.relative_to(output_dir)),
        },
        "counts": dict(sorted(counts.items())),
        "depth_distribution": dict(sorted(depths.items())),
        "mime_distribution": dict(sorted(mimes.items())),
        "missing_fields": dict(sorted(missing_fields.items())),
        "validation_issues": dict(sorted(validation_counts.items())),
        "metadata_cross_check": dict(sorted(metadata_counts.items())),
        "exclusion_reasons": dict(sorted(exclusion_counts.items())),
        "character_totals": {
            "source_text": source_chars,
            "retained_text": retained_chars,
        },
    }
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile


def process_all(input_dir: Path, selected_csv: Path, output_dir: Path) -> dict[str, Any]:
    """Process every top-level company JSONL and write an aggregate summary."""
    selected = load_selected_metadata(selected_csv)
    source_paths = sorted(input_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not source_paths:
        raise FileNotFoundError(f"No company JSONL files found in {input_dir}")

    profiles = [process_company(path, selected, output_dir) for path in source_paths]
    aggregate_counts: Counter[str] = Counter()
    aggregate_exclusions: Counter[str] = Counter()
    aggregate_validation: Counter[str] = Counter()
    for profile in profiles:
        aggregate_counts.update(profile["counts"])
        aggregate_exclusions.update(profile["exclusion_reasons"])
        aggregate_validation.update(profile["validation_issues"])

    summary = {
        "phase": "1-2",
        "source_directory": str(input_dir),
        "selected_metadata": str(selected_csv),
        "company_count": len(profiles),
        "companies": [profile["slug"] for profile in profiles],
        "counts": dict(sorted(aggregate_counts.items())),
        "exclusion_reasons": dict(sorted(aggregate_exclusions.items())),
        "validation_issues": dict(sorted(aggregate_validation.items())),
        "configuration": {
            "minimum_substantive_characters": MIN_SUBSTANTIVE_CHARS,
            "pdf_extraction": False,
            "text_modified": False,
            "token_counting": False,
            "deduplication": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
