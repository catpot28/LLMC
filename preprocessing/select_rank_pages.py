"""Phase 4 language selection and target-neutral ranking features.

Feature 4.1 extracts explicit language signals from URLs. It does not detect
language from page text, choose a company language, or filter pages.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import tiktoken
from lingua import Language, LanguageDetector, LanguageDetectorBuilder

from .clean_deduplicate import split_text_blocks


ROOT = Path(__file__).resolve().parent.parent


LANGUAGE_CODES = frozenset(
    "ar bg ca cs da de el en es et fi fr he hi hr hu id is it ja ko lt lv ms nl no "
    "pl pt ro ru sk sl sr sv th tr uk vi zh".split()
)

# ISO-like country/region codes commonly used in corporate locale URLs. They are
# used only to recognize the structure of locale pairs, not to infer a language.
COUNTRY_CODES = frozenset(
    "ae ar at au be bg br ca ch cl cn co cr cz de dk ee eg es fi fr gb gr hk hr hu "
    "id ie il in is it jp kr lt lu lv ma mx my nl no nz pe ph pk pl pt ro rs ru sa "
    "se sg si sk th tr tw ua uk us ve vn za".split()
)
# Additional country and regional namespaces observed on multinational sites.
# They are treated as URL structure only; a language is inferred solely when an
# explicit suffix such as _en, _ar, or _fil is present.
REGION_PATH_CODES = frozenset(
    "africa cis eastern-europe ec la mena ng pr sea".split()
)
REGION_LANGUAGE_RE = re.compile(r"^(.+?)[-_]([a-z]{2,3})$", re.IGNORECASE)
URL_LANGUAGE_SUFFIXES = LANGUAGE_CODES | frozenset({"fil"})

QUERY_LANGUAGE_KEYS = ("lang", "language", "locale", "hl")
LOCALE_RE = re.compile(r"^([a-z]{2})[-_]([a-z]{2})$", re.IGNORECASE)
# These codes can denote materially different languages and countries/regions in
# a first path segment (for example ca = Catalan or Canada, ar = Arabic or
# Argentina). URL evidence alone is insufficient in that context.
AMBIGUOUS_LANGUAGE_PATH_CODES = frozenset({"ar", "ca", "et", "sr", "sv", "uk"})
TEXT_LANGUAGE_MIN_CHARACTERS = 200
TEXT_LANGUAGE_MAX_ANALYSIS_CHARACTERS = 20_000
TEXT_LANGUAGE_MIN_CONFIDENCE = 0.75
TEXT_LANGUAGE_MIN_MARGIN = 0.20
ENGLISH_MINIMUM_ESTIMATED_TOKENS = 5_000
ESTIMATED_CHARACTERS_PER_TOKEN = 4
TARGET_MINIMUM_ESTIMATED_TOKENS = 45_000
TARGET_MAXIMUM_ESTIMATED_TOKENS = 47_000
DEFAULT_BUDGET_ENCODING = "o200k_base"
MAXIMUM_PAGE_CONTRIBUTION_TOKENS = 5_000

PAGE_TYPE_PRECEDENCE = (
    "homepage",
    "technical_documentation",
    "careers",
    "contact_locations",
    "cases_projects",
    "research_reports",
    "news_articles",
    "products_services",
    "industries_markets",
    "about_company",
)
PAGE_TYPE_URL_PATTERNS = {
    "technical_documentation": re.compile(
        r"(?:^|/)(?:docs?|documentation|developers?|api[-_]?reference|technical[-_]?documentation)(?:/|$)"
    ),
    "careers": re.compile(
        r"(?:^|/)(?:careers?|jobs?|vacancies|work[-_]?with[-_]?us|join[-_]?(?:us|our[-_]?team))(?:/|$)"
    ),
    "contact_locations": re.compile(
        r"(?:^|/)(?:contact(?:[-_]?us)?|locations?|offices?|stores?|find[-_]?us)(?:/|$)"
    ),
    "cases_projects": re.compile(
        r"(?:^|/)(?:case[-_]?(?:study|studies)|customer[-_]?(?:story|stories)|"
        r"success[-_]?(?:story|stories)|projects?|portfolio)(?:/|$)"
    ),
    "research_reports": re.compile(
        r"(?:^|/)(?:research|reports?|white[-_]?papers?|publications?|insights?)(?:/|$)"
    ),
    "news_articles": re.compile(
        r"(?:^|/)(?:news|blogs?|articles?|press|media|stories)(?:/|$)"
    ),
    "products_services": re.compile(
        r"(?:^|/)(?:products?|services?|solutions?|offerings?|capabilities)(?:/|$)"
    ),
    "industries_markets": re.compile(
        r"(?:^|/)(?:industries|industry|sectors?|markets?)(?:/|$)"
    ),
    "about_company": re.compile(
        r"(?:^|/)(?:about(?:[-_]?us)?|company|who[-_]?we[-_]?are|leadership|management|"
        r"history|mission|values|governance|investors?)(?:/|$)"
    ),
}
PAGE_TYPE_TITLE_PATTERNS = {
    "technical_documentation": re.compile(
        r"\b(?:documentation|developer docs|api reference|technical documentation)\b", re.I
    ),
    "careers": re.compile(
        r"\b(?:careers?|jobs?|vacancies|work with us|join (?:us|our team))\b", re.I
    ),
    "contact_locations": re.compile(
        r"\b(?:contact us|locations?|our offices?|find us|store locator)\b", re.I
    ),
    "cases_projects": re.compile(
        r"\b(?:case stud(?:y|ies)|customer stor(?:y|ies)|success stor(?:y|ies)|projects?|portfolio)\b",
        re.I,
    ),
    "research_reports": re.compile(
        r"\b(?:research|reports?|white ?papers?|publications?|insights?)\b", re.I
    ),
    "news_articles": re.compile(
        r"\b(?:news|blogs?|articles?|press releases?|media center|stories)\b", re.I
    ),
    "products_services": re.compile(
        r"\b(?:products?|services?|solutions?|offerings?|capabilities)\b", re.I
    ),
    "industries_markets": re.compile(
        r"\b(?:industries|industry|sectors?|markets?)\b", re.I
    ),
    "about_company": re.compile(
        r"\b(?:about us|our company|who we are|leadership|management|our history|"
        r"our mission|our values|governance|investor relations)\b",
        re.I,
    ),
}


@dataclass(frozen=True)
class URLLanguageSignal:
    """One explicit language indication extracted from a URL component."""

    language: str
    source: str
    raw_value: str
    confidence: float


@dataclass(frozen=True)
class URLLanguageInference:
    """Combined result of all explicit URL language signals."""

    language: str | None
    status: str
    confidence: float | None
    signals: tuple[URLLanguageSignal, ...]


@dataclass(frozen=True)
class TextLanguageInference:
    """Language result from cleaned page text."""

    language: str | None
    status: str
    confidence: float | None
    confidence_margin: float | None
    analyzed_characters: int


@dataclass(frozen=True)
class PageLanguageInference:
    """Resolved page language with both URL and cleaned-text evidence."""

    language: str | None
    status: str
    confidence: float | None
    url: URLLanguageInference
    text: TextLanguageInference


@dataclass(frozen=True)
class PageTypeInference:
    """Deterministic target-neutral page-type decision."""

    page_type: str
    source: str
    url_matches: tuple[str, ...]
    title_matches: tuple[str, ...]
    normalized_path: str


@dataclass(frozen=True)
class LocalePageIdentity:
    """Locale-neutral identity used to consolidate regional page variants."""

    identity: str
    normalized_host: str
    normalized_path: str
    removed_locale: str | None
    explicit_language: str | None


def _language_from_locale(value: str) -> str | None:
    """Resolve en-US, US-en, or a two-letter language code."""
    normalized = value.strip().lower().replace("_", "-")
    if normalized in LANGUAGE_CODES:
        return normalized
    match = LOCALE_RE.fullmatch(normalized)
    if match is None:
        return None
    first, second = match.groups()
    country_language = first in COUNTRY_CODES and second in LANGUAGE_CODES
    language_country = first in LANGUAGE_CODES and second in COUNTRY_CODES
    if country_language and not language_country:
        return second
    if language_country and not country_language:
        return first
    if country_language and language_country:
        # de-DE and similar symmetric locales resolve consistently. For genuinely
        # ambiguous pairs, corporate sites in this dataset predominantly use the
        # country-language convention (for example nl-en and ca-fr).
        return second if first != second else first
    return None


def infer_url_language(url: str) -> URLLanguageInference:
    """Extract explicit URL language signals without examining page content."""
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    try:
        parsed = urlsplit(url)
    except ValueError:
        return URLLanguageInference(None, "invalid_url", None, ())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return URLLanguageInference(None, "invalid_url", None, ())

    signals: list[URLLanguageSignal] = []

    query = parse_qs(parsed.query, keep_blank_values=False)
    lower_query = {key.lower(): values for key, values in query.items()}
    for key in QUERY_LANGUAGE_KEYS:
        for value in lower_query.get(key, []):
            language = _language_from_locale(value)
            if language is not None:
                signals.append(URLLanguageSignal(language, f"query:{key}", value, 1.0))

    segments = [unquote(segment).lower() for segment in parsed.path.split("/") if segment]
    if segments:
        first_segment = segments[0]
        locale_language = _language_from_locale(first_segment)
        if LOCALE_RE.fullmatch(first_segment) and locale_language is not None:
            signals.append(
                URLLanguageSignal(locale_language, "path_locale", first_segment, 0.98)
            )
        elif (
            first_segment in LANGUAGE_CODES
            and first_segment not in AMBIGUOUS_LANGUAGE_PATH_CODES
        ):
            signals.append(
                URLLanguageSignal(first_segment, "path_language", first_segment, 0.90)
            )

    hostname = (parsed.hostname or "").lower()
    labels = hostname.split(".")
    if len(labels) >= 3:
        first_label = labels[0]
        subdomain_language = _language_from_locale(first_label)
        if subdomain_language is not None and (
            first_label in LANGUAGE_CODES or LOCALE_RE.fullmatch(first_label)
        ):
            signals.append(
                URLLanguageSignal(subdomain_language, "locale_subdomain", first_label, 0.90)
            )

    ordered_signals = tuple(
        sorted(signals, key=lambda signal: (-signal.confidence, signal.source, signal.raw_value))
    )
    if not ordered_signals:
        return URLLanguageInference(None, "missing", None, ())
    languages = {signal.language for signal in ordered_signals}
    if len(languages) > 1:
        return URLLanguageInference(None, "conflict", None, ordered_signals)
    return URLLanguageInference(
        language=ordered_signals[0].language,
        status="detected",
        confidence=max(signal.confidence for signal in ordered_signals),
        signals=ordered_signals,
    )


def locale_neutral_page_identity(url: str) -> LocalePageIdentity:
    """Remove one leading locale/region namespace from a page URL."""
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    labels = host.split(".")
    explicit_language: str | None = None
    removed_locale: str | None = None
    if len(labels) >= 3:
        subdomain_language = _language_from_locale(labels[0])
        if subdomain_language is not None and (
            labels[0] in LANGUAGE_CODES or LOCALE_RE.fullmatch(labels[0])
        ):
            explicit_language = subdomain_language
            removed_locale = labels[0]
            host = ".".join(labels[1:])

    segments = [unquote(segment).strip().lower() for segment in parsed.path.split("/") if segment]
    if segments:
        first = segments[0]
        locale_language = _language_from_locale(first)
        region_language_match = REGION_LANGUAGE_RE.fullmatch(first)
        region_language = (
            region_language_match.group(2).lower()
            if region_language_match
            and region_language_match.group(2).lower() in URL_LANGUAGE_SUFFIXES
            else None
        )
        is_locale = bool(LOCALE_RE.fullmatch(first)) or first in LANGUAGE_CODES
        is_region = first in COUNTRY_CODES or first in REGION_PATH_CODES
        if is_locale or is_region or region_language is not None:
            removed_locale = first
            if region_language is not None:
                explicit_language = region_language
            elif is_locale and locale_language is not None:
                explicit_language = locale_language
            segments = segments[1:]
    normalized_path = "/" + "/".join(segments) if segments else "/"
    return LocalePageIdentity(
        identity=f"{host}{normalized_path}",
        normalized_host=host,
        normalized_path=normalized_path,
        removed_locale=removed_locale,
        explicit_language=explicit_language,
    )


@lru_cache(maxsize=1)
def _text_language_detector() -> LanguageDetector:
    """Build one deterministic detector shared by all processed pages."""
    supported = [
        language
        for language in Language.all()
        if _iso_language_code(language) in LANGUAGE_CODES
    ]
    return LanguageDetectorBuilder.from_languages(*supported).build()


def _iso_language_code(language: Language) -> str:
    code = language.iso_code_639_1.name.lower()
    # Lingua distinguishes the two written Norwegian standards. The pipeline's
    # company-level policy uses their shared ISO 639-1 language code.
    return "no" if code in {"nb", "nn"} else code


def _representative_text(text: str) -> str:
    """Bound runtime while sampling the beginning, middle, and end of long pages."""
    if len(text) <= TEXT_LANGUAGE_MAX_ANALYSIS_CHARACTERS:
        return text
    third = TEXT_LANGUAGE_MAX_ANALYSIS_CHARACTERS // 3
    middle_start = max(0, len(text) // 2 - third // 2)
    return "\n".join((text[:third], text[middle_start : middle_start + third], text[-third:]))


def infer_text_language(text: str) -> TextLanguageInference:
    """Detect language from cleaned text and reject weak results."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    substantive = "".join(character for character in text if not character.isspace())
    if len(substantive) < TEXT_LANGUAGE_MIN_CHARACTERS:
        return TextLanguageInference(None, "insufficient_text", None, None, len(text))

    analyzed = _representative_text(text)
    values = _text_language_detector().compute_language_confidence_values(analyzed)
    if not values:
        return TextLanguageInference(None, "undetected", None, None, len(analyzed))
    top = values[0]
    second_confidence = values[1].value if len(values) > 1 else 0.0
    confidence = float(top.value)
    margin = confidence - float(second_confidence)
    language = _iso_language_code(top.language)
    if confidence < TEXT_LANGUAGE_MIN_CONFIDENCE or margin < TEXT_LANGUAGE_MIN_MARGIN:
        return TextLanguageInference(None, "low_confidence", confidence, margin, len(analyzed))
    return TextLanguageInference(language, "detected", confidence, margin, len(analyzed))


def infer_page_language(url: str, text: str) -> PageLanguageInference:
    """Resolve page language using cleaned text as authority when confident."""
    url_result = infer_url_language(url)
    text_result = infer_text_language(text)
    text_language = text_result.language
    url_language = url_result.language

    if text_language is not None:
        if url_result.status == "conflict":
            status = "text_resolved_url_conflict"
        elif url_language is None:
            status = "text_only"
        elif url_language == text_language:
            status = "agreement"
        else:
            status = "conflict_text_selected"
        return PageLanguageInference(
            text_language, status, text_result.confidence, url_result, text_result
        )

    if url_language is not None:
        return PageLanguageInference(
            url_language, "url_only", url_result.confidence, url_result, text_result
        )
    return PageLanguageInference(None, "unresolved", None, url_result, text_result)


def _page_type_path(url: str) -> tuple[list[str], str]:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return [], ""
    segments = [unquote(segment).strip().lower() for segment in parsed.path.split("/") if segment]
    if segments:
        first = segments[0]
        if (
            LOCALE_RE.fullmatch(first)
            or first in LANGUAGE_CODES
            or first in COUNTRY_CODES
        ):
            segments = segments[1:]
    return segments, "/".join(segments)


def infer_page_type(url: str, title: str | None) -> PageTypeInference:
    """Classify generic page purpose using URL first and title second."""
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")
    segments, normalized_path = _page_type_path(url)
    if not segments or (len(segments) == 1 and segments[0] in {"home", "index", "index.html"}):
        return PageTypeInference("homepage", "url", ("homepage",), (), normalized_path)

    url_matches = tuple(
        page_type
        for page_type in PAGE_TYPE_PRECEDENCE
        if page_type != "homepage" and PAGE_TYPE_URL_PATTERNS[page_type].search(normalized_path)
    )
    compact_title = " ".join((title or "").split())
    title_matches = tuple(
        page_type
        for page_type in PAGE_TYPE_PRECEDENCE
        if page_type != "homepage" and PAGE_TYPE_TITLE_PATTERNS[page_type].search(compact_title)
    )
    if url_matches:
        return PageTypeInference(url_matches[0], "url", url_matches, title_matches, normalized_path)
    if title_matches:
        return PageTypeInference(title_matches[0], "title", url_matches, title_matches, normalized_path)
    return PageTypeInference("other", "default", url_matches, title_matches, normalized_path)


def estimate_language_tokens(text: str) -> int:
    """Return the documented Phase 4 character-based token estimate."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return ceil(len(text) / ESTIMATED_CHARACTERS_PER_TOKEN)


@lru_cache(maxsize=None)
def _budget_encoding(encoding_name: str):
    """Load a named tokenizer once so budget counts are reproducible."""
    return tiktoken.get_encoding(encoding_name)


def count_budget_tokens(text: str, encoding_name: str = DEFAULT_BUDGET_ENCODING) -> int:
    """Count text tokens with the explicitly recorded reference encoding."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(_budget_encoding(encoding_name).encode(text))


def page_budget_metadata(
    text: str,
    encoding_name: str = DEFAULT_BUDGET_ENCODING,
    maximum_page_tokens: int = MAXIMUM_PAGE_CONTRIBUTION_TOKENS,
) -> dict[str, Any]:
    """Describe whole-page and block-level candidates without selecting them."""
    if maximum_page_tokens <= 0:
        raise ValueError("maximum_page_tokens must be positive")
    page_tokens = count_budget_tokens(text, encoding_name)
    blocks = split_text_blocks(text)
    block_candidates = [
        {
            "block_index": block.index,
            "text": block.text,
            "character_count": block.character_count,
            "token_count": count_budget_tokens(block.text, encoding_name),
        }
        for block in blocks
    ]
    return {
        "tokenizer": encoding_name,
        "page_token_count": page_tokens,
        "maximum_page_contribution_tokens": maximum_page_tokens,
        "oversized": page_tokens > maximum_page_tokens,
        "candidate_mode": "blocks" if page_tokens > maximum_page_tokens else "whole_page",
        "block_count": len(block_candidates),
        "block_candidates": block_candidates if page_tokens > maximum_page_tokens else [],
        "selection_performed": False,
        "selected_token_count": None,
    }


def choose_company_language(language_tokens: dict[str, int]) -> tuple[str | None, str]:
    """Choose English when sufficiently available, otherwise one fallback language."""
    english_tokens = language_tokens.get("en", 0)
    if english_tokens >= ENGLISH_MINIMUM_ESTIMATED_TOKENS:
        return "en", "english_threshold_met"

    non_english = [
        (tokens, language)
        for language, tokens in language_tokens.items()
        if language != "en" and tokens > 0
    ]
    if non_english:
        tokens, language = max(non_english, key=lambda item: (item[0], item[1]))
        return language, "dominant_non_english_fallback"
    if english_tokens > 0:
        return "en", "english_only_below_threshold"
    return None, "no_resolved_language"


def _write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _language_audit(result: PageLanguageInference) -> dict[str, Any]:
    return {
        "final_language": result.language,
        "resolution_status": result.status,
        "confidence": result.confidence,
        "url": asdict(result.url),
        "text": asdict(result.text),
    }


def process_language_company(source_path: Path, output_dir: Path) -> dict[str, Any]:
    """Resolve page languages and write one selected language for a company."""
    records: list[dict[str, Any]] = []
    inferences: list[PageLanguageInference] = []
    token_totals: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{source_path}:{line_number} must contain a JSON object")
            url = record.get("url")
            text = record.get("text")
            if not isinstance(url, str) or not isinstance(text, str):
                raise TypeError(f"{source_path}:{line_number} requires string URL and text")
            result = infer_page_language(url, text)
            records.append(record)
            inferences.append(result)
            status_counts[result.status] += 1
            if result.language is not None:
                token_totals[result.language] += estimate_language_tokens(text)

    selected_language, selection_reason = choose_company_language(dict(token_totals))
    slug = source_path.stem
    selected_path = output_dir / "language_selected" / source_path.name
    excluded_path = output_dir / "language_excluded" / source_path.name
    profile_path = output_dir / "profiles" / f"{slug}.json"
    for path in (selected_path, excluded_path, profile_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    selected_pages = 0
    excluded_pages = 0
    unresolved_pages = 0
    selected_characters = 0
    selected_estimated_tokens = 0
    with (
        selected_path.open("w", encoding="utf-8") as selected_output,
        excluded_path.open("w", encoding="utf-8") as excluded_output,
    ):
        for page_index, (record, result) in enumerate(zip(records, inferences)):
            language_metadata = _language_audit(result)
            estimated_tokens = estimate_language_tokens(record["text"])
            if selected_language is not None and result.language == selected_language:
                output_record = dict(record)
                output_record["_phase4"] = {
                    "status": "language_selected",
                    "selected_company_language": selected_language,
                    "company_selection_reason": selection_reason,
                    "estimated_tokens": estimated_tokens,
                    "language": language_metadata,
                }
                _write_jsonl_row(selected_output, output_record)
                selected_pages += 1
                selected_characters += len(record["text"])
                selected_estimated_tokens += estimated_tokens
                continue

            primary_reason = (
                "unresolved_language"
                if result.language is None
                else "non_selected_language"
            )
            if result.language is None:
                unresolved_pages += 1
            audit = {
                "page_index": page_index,
                "status": "excluded",
                "primary_reason": primary_reason,
                "company": record.get("company"),
                "slug": record.get("slug"),
                "url": record.get("url"),
                "title": record.get("title"),
                "depth": record.get("depth"),
                "text_characters": len(record["text"]),
                "estimated_tokens": estimated_tokens,
                "selected_company_language": selected_language,
                "language": language_metadata,
            }
            _write_jsonl_row(excluded_output, audit)
            excluded_pages += 1

    profile: dict[str, Any] = {
        "slug": slug,
        "source_file": str(source_path),
        "selected_language": selected_language,
        "selection_reason": selection_reason,
        "language_estimated_tokens": dict(sorted(token_totals.items())),
        "resolution_statuses": dict(sorted(status_counts.items())),
        "counts": {
            "phase3_pages": len(records),
            "selected_pages": selected_pages,
            "excluded_pages": excluded_pages,
            "unresolved_pages": unresolved_pages,
        },
        "selected_text": {
            "characters": selected_characters,
            "estimated_tokens": selected_estimated_tokens,
        },
        "output_files": {
            "selected": str(selected_path.relative_to(output_dir)),
            "excluded": str(excluded_path.relative_to(output_dir)),
        },
    }
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile


def process_language_all(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Apply English-first language selection to all Phase 3 companies."""
    source_paths = sorted(input_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not source_paths:
        raise FileNotFoundError(f"No Phase 3 company JSONL files found in {input_dir}")

    profiles = [process_language_company(path, output_dir) for path in source_paths]
    counts: Counter[str] = Counter()
    selected_languages: Counter[str] = Counter()
    selection_reasons: Counter[str] = Counter()
    selected_characters = 0
    selected_estimated_tokens = 0
    for profile in profiles:
        counts.update(profile["counts"])
        if profile["selected_language"] is not None:
            selected_languages[profile["selected_language"]] += 1
        selection_reasons[profile["selection_reason"]] += 1
        selected_characters += profile["selected_text"]["characters"]
        selected_estimated_tokens += profile["selected_text"]["estimated_tokens"]

    summary = {
        "phase": "4-language-selection",
        "company_count": len(profiles),
        "counts": dict(sorted(counts.items())),
        "selected_company_languages": dict(sorted(selected_languages.items())),
        "selection_reasons": dict(sorted(selection_reasons.items())),
        "selected_text": {
            "characters": selected_characters,
            "estimated_tokens": selected_estimated_tokens,
        },
        "configuration": {
            "english_minimum_estimated_tokens": ENGLISH_MINIMUM_ESTIMATED_TOKENS,
            "estimated_characters_per_token": ESTIMATED_CHARACTERS_PER_TOKEN,
            "text_minimum_characters": TEXT_LANGUAGE_MIN_CHARACTERS,
            "text_minimum_confidence": TEXT_LANGUAGE_MIN_CONFIDENCE,
            "text_minimum_confidence_margin": TEXT_LANGUAGE_MIN_MARGIN,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "language_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _locale_canonical_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Prefer shallow, substantive, compact, and finally lexical variants."""
    depth = record.get("depth")
    stable_depth = depth if isinstance(depth, int) and not isinstance(depth, bool) else 10**9
    text = record.get("text", "")
    substantive = sum(not character.isspace() for character in text)
    url = str(record.get("url", ""))
    return stable_depth, -substantive, len(url), url


def process_locale_company(source_path: Path, output_dir: Path) -> dict[str, Any]:
    """Keep one representative for each locale-neutral logical page."""
    records = _load_locale_records(source_path)
    slug = source_path.stem
    retained_path = output_dir / "locale_consolidated" / source_path.name
    excluded_path = output_dir / "locale_excluded" / source_path.name
    profile_path = output_dir / "locale_profiles" / f"{slug}.json"
    for path in (retained_path, excluded_path, profile_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    eligible: list[tuple[int, dict[str, Any], LocalePageIdentity]] = []
    excluded: list[dict[str, Any]] = []
    explicit_mismatch_pages = 0
    for index, record in enumerate(records):
        selected_language = record.get("_phase4", {}).get("selected_company_language")
        identity = locale_neutral_page_identity(record["url"])
        if (
            identity.explicit_language is not None
            and selected_language is not None
            and identity.explicit_language != selected_language
        ):
            explicit_mismatch_pages += 1
            excluded.append({
                "page_index": index,
                "status": "excluded",
                "primary_reason": "explicit_url_language_mismatch",
                "url": record.get("url"),
                "title": record.get("title"),
                "depth": record.get("depth"),
                "selected_company_language": selected_language,
                "url_explicit_language": identity.explicit_language,
                "locale_page_identity": asdict(identity),
            })
            continue
        eligible.append((index, record, identity))

    groups: dict[str, list[tuple[int, dict[str, Any], LocalePageIdentity]]] = {}
    for item in eligible:
        groups.setdefault(item[2].identity, []).append(item)

    retained: list[tuple[int, dict[str, Any]]] = []
    consolidated_variants = 0
    variant_groups = 0
    for identity_key in sorted(groups):
        members = groups[identity_key]
        canonical = min(members, key=lambda item: _locale_canonical_key(item[1]))
        canonical_index, canonical_record, identity = canonical
        output_record = dict(canonical_record)
        phase4 = dict(output_record.get("_phase4", {}))
        phase4["locale_consolidation"] = {
            "status": "canonical",
            "locale_page_identity": asdict(identity),
            "variant_count": len(members),
            "canonical_rule": [
                "lower_depth",
                "more_substantive_text",
                "shorter_url",
                "lexical_url",
            ],
        }
        output_record["_phase4"] = phase4
        retained.append((canonical_index, output_record))
        if len(members) > 1:
            variant_groups += 1
        for member_index, member_record, member_identity in members:
            if member_index == canonical_index:
                continue
            consolidated_variants += 1
            excluded.append({
                "page_index": member_index,
                "status": "excluded",
                "primary_reason": "locale_variant_duplicate",
                "url": member_record.get("url"),
                "title": member_record.get("title"),
                "depth": member_record.get("depth"),
                "locale_page_identity": asdict(member_identity),
                "canonical_url": canonical_record.get("url"),
            })

    with retained_path.open("w", encoding="utf-8") as handle:
        for _, record in sorted(retained, key=lambda item: item[0]):
            _write_jsonl_row(handle, record)
    with excluded_path.open("w", encoding="utf-8") as handle:
        for audit in sorted(excluded, key=lambda item: item["page_index"]):
            _write_jsonl_row(handle, audit)

    profile = {
        "slug": slug,
        "counts": {
            "language_selected_pages": len(records),
            "retained_pages": len(retained),
            "excluded_pages": len(excluded),
            "explicit_language_mismatch_pages": explicit_mismatch_pages,
            "locale_variant_duplicate_pages": consolidated_variants,
            "locale_variant_groups": variant_groups,
        },
        "output_files": {
            "retained": str(retained_path.relative_to(output_dir)),
            "excluded": str(excluded_path.relative_to(output_dir)),
        },
    }
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile


def _load_locale_records(source_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("url"), str)
                or not isinstance(record.get("text"), str)
            ):
                raise TypeError(f"{source_path}:{line_number} requires string URL and text")
            records.append(record)
    return records


def process_locale_all(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Consolidate regional variants for every selected-language company."""
    source_paths = sorted(input_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not source_paths:
        raise FileNotFoundError(f"No language-selected files found in {input_dir}")
    profiles = [process_locale_company(path, output_dir) for path in source_paths]
    counts: Counter[str] = Counter()
    for profile in profiles:
        counts.update(profile["counts"])
    summary = {
        "phase": "4-locale-consolidation",
        "company_count": len(profiles),
        "counts": dict(sorted(counts.items())),
        "policy": {
            "one_representative_per_locale_neutral_page_identity": True,
            "explicit_url_language_mismatch_excluded": True,
            "body_language_cannot_override_explicit_url_mismatch": True,
            "target_neutral": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "locale_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def depth_pool_status(primary_tokens: int, reserve_tokens: int) -> str:
    """Classify how a company must proceed toward the token target."""
    if primary_tokens > TARGET_MAXIMUM_ESTIMATED_TOKENS:
        return "needs_primary_selection"
    if primary_tokens >= TARGET_MINIMUM_ESTIMATED_TOKENS:
        return "primary_ready"
    if reserve_tokens > 0:
        return "needs_depth3_fill"
    return "underfilled_no_reserve"


def process_depth_company(source_path: Path, output_dir: Path) -> dict[str, Any]:
    """Partition selected-language pages into shallow primary and depth-3 reserve pools."""
    slug = source_path.stem
    primary_path = output_dir / "depth_primary" / source_path.name
    reserve_path = output_dir / "depth3_reserve" / source_path.name
    other_path = output_dir / "depth_other" / source_path.name
    profile_path = output_dir / "depth_profiles" / f"{slug}.json"
    for path in (primary_path, reserve_path, other_path, profile_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    characters: Counter[str] = Counter()
    with (
        source_path.open(encoding="utf-8") as source,
        primary_path.open("w", encoding="utf-8") as primary_output,
        reserve_path.open("w", encoding="utf-8") as reserve_output,
        other_path.open("w", encoding="utf-8") as other_output,
    ):
        for page_index, line in enumerate(source):
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise TypeError(f"{source_path}:{page_index + 1} must contain page text")
            depth = record.get("depth")
            page_tokens = estimate_language_tokens(record["text"])
            page_characters = len(record["text"])
            output_record = dict(record)

            if isinstance(depth, int) and not isinstance(depth, bool) and 0 <= depth <= 2:
                pool = "primary_depth_0_2"
                destination = primary_output
                counter_key = "primary"
            elif depth == 3:
                pool = "depth_3_reserve"
                destination = reserve_output
                counter_key = "reserve"
            else:
                pool = "unexpected_depth"
                destination = other_output
                counter_key = "other"

            phase4 = dict(output_record.get("_phase4", {}))
            phase4["depth_pool"] = {
                "pool": pool,
                "depth": depth,
                "estimated_tokens": page_tokens,
            }
            output_record["_phase4"] = phase4
            _write_jsonl_row(destination, output_record)
            counts[f"{counter_key}_pages"] += 1
            tokens[f"{counter_key}_estimated_tokens"] += page_tokens
            characters[f"{counter_key}_characters"] += page_characters

    primary_tokens = tokens["primary_estimated_tokens"]
    reserve_tokens = tokens["reserve_estimated_tokens"]
    status = depth_pool_status(primary_tokens, reserve_tokens)
    profile: dict[str, Any] = {
        "slug": slug,
        "status": status,
        "counts": dict(sorted(counts.items())),
        "estimated_tokens": dict(sorted(tokens.items())),
        "characters": dict(sorted(characters.items())),
        "target": {
            "minimum_estimated_tokens": TARGET_MINIMUM_ESTIMATED_TOKENS,
            "maximum_estimated_tokens": TARGET_MAXIMUM_ESTIMATED_TOKENS,
        },
        "output_files": {
            "primary": str(primary_path.relative_to(output_dir)),
            "reserve": str(reserve_path.relative_to(output_dir)),
            "other": str(other_path.relative_to(output_dir)),
        },
    }
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile


def process_depth_all(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create depth candidate pools for all selected-language company files."""
    source_paths = sorted(input_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not source_paths:
        raise FileNotFoundError(f"No selected-language JSONL files found in {input_dir}")
    profiles = [process_depth_company(path, output_dir) for path in source_paths]

    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for profile in profiles:
        counts.update(profile["counts"])
        tokens.update(profile["estimated_tokens"])
        statuses[profile["status"]] += 1
    summary = {
        "phase": "4-depth-pools",
        "company_count": len(profiles),
        "statuses": dict(sorted(statuses.items())),
        "counts": dict(sorted(counts.items())),
        "estimated_tokens": dict(sorted(tokens.items())),
        "target": {
            "minimum_estimated_tokens": TARGET_MINIMUM_ESTIMATED_TOKENS,
            "maximum_estimated_tokens": TARGET_MAXIMUM_ESTIMATED_TOKENS,
        },
        "policy": {
            "primary_depths": [0, 1, 2],
            "reserve_depth": 3,
            "reserve_requires_ranking": True,
            "depth3_selected_at_this_stage": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def process_page_type_company(
    primary_path: Path, reserve_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Annotate one company's primary and reserve pools with neutral page types."""
    slug = primary_path.stem
    typed_primary_path = output_dir / "typed_primary" / primary_path.name
    typed_reserve_path = output_dir / "typed_depth3_reserve" / reserve_path.name
    profile_path = output_dir / "page_type_profiles" / f"{slug}.json"
    for path in (typed_primary_path, typed_reserve_path, profile_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    pool_type_counts: dict[str, Counter[str]] = {
        "primary": Counter(),
        "reserve": Counter(),
    }
    source_counts: Counter[str] = Counter()
    multiple_match_pages = 0
    for pool, source_path, output_path in (
        ("primary", primary_path, typed_primary_path),
        ("reserve", reserve_path, typed_reserve_path),
    ):
        with source_path.open(encoding="utf-8") as source, output_path.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("url"), str):
                    raise TypeError(f"{source_path}:{line_number} requires a string URL")
                result = infer_page_type(record["url"], record.get("title"))
                output_record = dict(record)
                phase4 = dict(output_record.get("_phase4", {}))
                phase4["page_type"] = asdict(result)
                output_record["_phase4"] = phase4
                _write_jsonl_row(destination, output_record)
                pool_type_counts[pool][result.page_type] += 1
                source_counts[result.source] += 1
                if len(set(result.url_matches + result.title_matches)) > 1:
                    multiple_match_pages += 1

    profile: dict[str, Any] = {
        "slug": slug,
        "page_types": {
            pool: dict(sorted(counts.items())) for pool, counts in pool_type_counts.items()
        },
        "decision_sources": dict(sorted(source_counts.items())),
        "multiple_match_pages": multiple_match_pages,
        "output_files": {
            "primary": str(typed_primary_path.relative_to(output_dir)),
            "reserve": str(typed_reserve_path.relative_to(output_dir)),
        },
    }
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile


def process_page_types_all(
    primary_dir: Path, reserve_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Annotate all depth pools with target-neutral page types."""
    primary_paths = sorted(primary_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not primary_paths:
        raise FileNotFoundError(f"No primary depth files found in {primary_dir}")
    profiles = []
    for primary_path in primary_paths:
        reserve_path = reserve_dir / primary_path.name
        if not reserve_path.exists():
            raise FileNotFoundError(f"Missing matching reserve file: {reserve_path}")
        profiles.append(process_page_type_company(primary_path, reserve_path, output_dir))

    primary_types: Counter[str] = Counter()
    reserve_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    multiple_matches = 0
    for profile in profiles:
        primary_types.update(profile["page_types"]["primary"])
        reserve_types.update(profile["page_types"]["reserve"])
        sources.update(profile["decision_sources"])
        multiple_matches += profile["multiple_match_pages"]
    summary = {
        "phase": "4-page-types",
        "company_count": len(profiles),
        "primary_page_types": dict(sorted(primary_types.items())),
        "reserve_page_types": dict(sorted(reserve_types.items())),
        "decision_sources": dict(sorted(sources.items())),
        "multiple_match_pages": multiple_matches,
        "precedence": list(PAGE_TYPE_PRECEDENCE) + ["other"],
        "target_neutral": True,
        "body_text_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "page_type_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def process_budget_candidates_company(
    primary_path: Path,
    reserve_path: Path,
    output_dir: Path,
    encoding_name: str = DEFAULT_BUDGET_ENCODING,
    maximum_page_tokens: int = MAXIMUM_PAGE_CONTRIBUTION_TOKENS,
) -> dict[str, Any]:
    """Annotate one company's typed pools for later budgeted selection."""
    slug = primary_path.stem
    candidate_primary_path = output_dir / "budget_primary" / primary_path.name
    candidate_reserve_path = output_dir / "budget_depth3_reserve" / reserve_path.name
    profile_path = output_dir / "budget_profiles" / f"{slug}.json"
    for path in (candidate_primary_path, candidate_reserve_path, profile_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    pool_counts: dict[str, Counter[str]] = {"primary": Counter(), "reserve": Counter()}
    for pool, source_path, destination_path in (
        ("primary", primary_path, candidate_primary_path),
        ("reserve", reserve_path, candidate_reserve_path),
    ):
        with source_path.open(encoding="utf-8") as source, destination_path.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise TypeError(f"{source_path}:{line_number} requires string page text")
                budget = page_budget_metadata(
                    record["text"], encoding_name, maximum_page_tokens
                )
                output_record = dict(record)
                phase4 = dict(output_record.get("_phase4", {}))
                phase4["budget_candidate"] = budget
                output_record["_phase4"] = phase4
                _write_jsonl_row(destination, output_record)

                pool_counts[pool]["pages"] += 1
                pool_counts[pool]["page_tokens"] += budget["page_token_count"]
                if budget["oversized"]:
                    pool_counts[pool]["oversized_pages"] += 1
                    pool_counts[pool]["oversized_page_tokens"] += budget[
                        "page_token_count"
                    ]
                    pool_counts[pool]["block_candidates"] += budget["block_count"]

    profile: dict[str, Any] = {
        "slug": slug,
        "tokenizer": encoding_name,
        "maximum_page_contribution_tokens": maximum_page_tokens,
        "pools": {
            pool: dict(sorted(counts.items())) for pool, counts in pool_counts.items()
        },
        "selection_performed": False,
        "output_files": {
            "primary": str(candidate_primary_path.relative_to(output_dir)),
            "reserve": str(candidate_reserve_path.relative_to(output_dir)),
        },
    }
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profile


def process_budget_candidates_all(
    primary_dir: Path,
    reserve_dir: Path,
    output_dir: Path,
    encoding_name: str = DEFAULT_BUDGET_ENCODING,
    maximum_page_tokens: int = MAXIMUM_PAGE_CONTRIBUTION_TOKENS,
) -> dict[str, Any]:
    """Prepare auditable whole-page/block candidates; do not rank or select."""
    primary_paths = sorted(primary_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not primary_paths:
        raise FileNotFoundError(f"No typed primary files found in {primary_dir}")
    profiles = []
    for primary_path in primary_paths:
        reserve_path = reserve_dir / primary_path.name
        if not reserve_path.exists():
            raise FileNotFoundError(f"Missing matching typed reserve file: {reserve_path}")
        profiles.append(
            process_budget_candidates_company(
                primary_path,
                reserve_path,
                output_dir,
                encoding_name,
                maximum_page_tokens,
            )
        )

    aggregate: dict[str, Counter[str]] = {"primary": Counter(), "reserve": Counter()}
    for profile in profiles:
        for pool in aggregate:
            aggregate[pool].update(profile["pools"][pool])
    summary = {
        "phase": "4-budget-candidates",
        "company_count": len(profiles),
        "tokenizer": encoding_name,
        "tokenizer_scope": "reference tokenizer for cross-model budgeting",
        "maximum_page_contribution_tokens": maximum_page_tokens,
        "pools": {
            pool: dict(sorted(counts.items())) for pool, counts in aggregate.items()
        },
        "policy": {
            "oversized_when_page_tokens_exceed": maximum_page_tokens,
            "oversized_pages_become_block_candidates": True,
            "whole_pages_excluded_for_length": False,
            "ranking_performed": False,
            "selection_performed": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "budget_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
