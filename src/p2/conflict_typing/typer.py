"""
Rule-based conflict typer — news-aware.

Design choice: transparent, deterministic rules rather than a trained
classifier. Reasons:
  1. No labeled conflict-type data exists on AVeriTeC (only verdict labels).
  2. A rule-based router is debuggable and inspectable by rationale,
     which matters for an explanation-first RAG system.
  3. When we later train a classifier, it can inherit type_pair's
     signature and drop in.

News-domain adaptations (v0.3):
  - Bilingual (EN/中文) opinion + reported-speech cues.
  - Month-precision temporal comparison, with claim_date awareness.
  - Source-medium authority tiers aligned to news industry
    (wire service > newspaper/TV > tabloid > blog/social).
    - Agreement/neutral rescue path using lexical contradiction cues
        and source-quality gaps when NLI is missing or uninformative.
  - `op_ed / editorial / commentary` medium routes to opinion_conflict,
    NOT misinformation — an op-ed is subjective, not low-quality.
  - Numerical-clash detector: different stats for the same subject
    route to ambiguity (or temporal if years also differ).
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse
from typing import Dict, List, Optional, Set, Tuple

from ..contracts import StancedPair
from ..p1_adapter import Claim, InputRecord
from .contracts import (
    TypedPair,
    TypedSample,
    DEFAULT_POLICY_BY_TYPE,
)


# Shared thresholds. Keeping them centralized makes ablation easier.
_AUTHORITY_GAP_MIN = 0.35
_SOURCE_GAP_OVERRIDE_MIN = 0.55
_NUMERICAL_REL_DIFF_MIN = 0.1
_NEGATION_OVERRIDE_MIN_OVERLAP = 4
_REFUTATION_OVERRIDE_MIN_OVERLAP = 2


# ---------------------------------------------------------------------------
# 1. Opinion / reported-speech cues (bilingual)
# ---------------------------------------------------------------------------

_OPINION_PATTERNS = [
    # English: belief / argument verbs. NOTE: "claim/claims" deliberately
    # excluded — too often used as a generic noun in news text (and in our
    # own fixtures) to be a reliable opinion signal.
    r"\b(believe[sd]?|think[s]?|argue[sd]?|assert[s]?|asserted|"
    r"contend[s]?|contended|maintain[s]?|suggested?)\b",
    # English: reported speech (news register)
    r"\b(said|says|told|stated|announced|declared|remarked|quoted|"
    r"according\s+to|reportedly|allegedly|allege[sd]?)\b",
    r"\b(spokes(?:person|man|woman)|sources?\s+say|experts?\s+say|"
    r"analysts?\s+say|officials?\s+said)\b",
    # English: opinion packaging
    r"\b(in\s+my\s+opinion|op[- ]ed|editorial|commentary|column(?:ist)?)\b",
    # Chinese: belief / assertion (no \b — Chinese has no word boundary)
    r"(认为|觉得|相信|主张|指出|表示|声称|宣称|断言|强调)",
    # Chinese: reported speech (news register)
    r"(据称|据悉|据报道|据了解|消息人士|知情人士|官员(?:称|表示)|"
    r"发言人|接受采访时)",
    # Chinese: opinion packaging
    r"(社论|评论员|观点|看法|立场|专栏)",
]
_OPINION_RE = re.compile("|".join(_OPINION_PATTERNS), flags=re.IGNORECASE)


def _has_opinion_marker(claim: Claim) -> bool:
    return bool(_OPINION_RE.search(claim.text or ""))


# ---------------------------------------------------------------------------
# 2. Source-medium authority tiers (news-industry aligned)
# ---------------------------------------------------------------------------

# Opinion-packaging mediums — NOT scored as low authority. They route
# the pair to opinion_conflict regardless of the text content.
_OPINION_MEDIA = frozenset({
    "op_ed", "opinion", "editorial", "commentary", "column",
})

_MEDIUM_ALIASES = {
    "op-ed": "op_ed",
    "web_text": "news",
    "webtable": "data_table",
    "web_table": "data_table",
    "image_graphic": "image",
    "graphic": "image",
    "pdf": "document",
}

_REPORTING_SOURCE_SOCIAL_RE = re.compile(
    r"facebook|twitter|tweeter|instagram|youtube|tiktok|whatsapp|social\s*media",
    flags=re.IGNORECASE,
)
_REPORTING_SOURCE_BLOG_RE = re.compile(r"blog|forum|reddit", flags=re.IGNORECASE)
_REPORTING_SOURCE_GOV_RE = re.compile(
    r"government|ministry|official|state\s*media|press\s*release",
    flags=re.IGNORECASE,
)
_REPORTING_SOURCE_NEWS_RE = re.compile(
    r"news|reuters|cnn|bbc|channel|television|newspaper|media",
    flags=re.IGNORECASE,
)

_OPINION_URL_RE = re.compile(
    r"/(opinion|op-ed|editorial|commentary|columns?)/",
    flags=re.IGNORECASE,
)
_ARCHIVE_EMBEDDED_URL_RE = re.compile(
    r"/web/\d+/(https?://[^\s]+)",
    flags=re.IGNORECASE,
)

_LOW_AUTH_DOMAIN_SUFFIXES = (
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "telegram.me",
    "telegram.org",
    "rumble.com",
    "blogspot.com",
    "wordpress.com",
    "medium.com",
    "substack.com",
)
_HIGH_AUTH_DOMAIN_SUFFIXES = (
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "npr.org",
    "cnbc.com",
    "abcnews.go.com",
    "nbcnews.com",
    "edition.cnn.com",
    "transcripts.cnn.com",
    "cdc.gov",
    "who.int",
    "congress.gov",
    "un.org",
    "europa.eu",
    "politifact.com",
    "factcheck.org",
    "snopes.com",
    "africacheck.org",
    "aap.com.au",
)

# Authority tiers. Scores are rough; only the *gap* between two sides
# matters for routing.
_MEDIUM_AUTHORITY = {
    # Highest: primary / authoritative source
    "wire_service": 0.95, "news_agency": 0.95,
    "official": 0.9, "government": 0.9, "press_release": 0.88,
    "academic": 0.9, "peer_reviewed": 0.95, "journal": 0.9,

    # Mid-high: established mainstream
    "newspaper": 0.8, "press": 0.75, "tv_news": 0.75, "magazine": 0.7,

    # Mid: generic news bucket
    "news": 0.55,
    "document": 0.72,
    "data_table": 0.65,
    "image": 0.45,
    "video": 0.45,
    "metadata": 0.5,

    # Lower: tabloid-like / partisan
    "tabloid": 0.35, "partisan": 0.35,

    # Low: user-generated / unmoderated
    "blog": 0.2, "forum": 0.2,
    "social_media": 0.12, "user_generated": 0.12, "tweet": 0.12,
}


def _normalize_medium(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if not token:
        return ""
    token = token.replace("-", "_").replace("/", "_")
    token = re.sub(r"\s+", "_", token)
    return _MEDIUM_ALIASES.get(token, token)


def _infer_medium_from_reporting_source(reporting_source: str) -> str:
    text = str(reporting_source or "").strip().lower()
    if not text:
        return ""
    if _REPORTING_SOURCE_SOCIAL_RE.search(text):
        return "social_media"
    if _REPORTING_SOURCE_BLOG_RE.search(text):
        return "blog"
    if _REPORTING_SOURCE_GOV_RE.search(text):
        return "government"
    if _REPORTING_SOURCE_NEWS_RE.search(text):
        return "news"
    return ""


def _source_url_of(claim: Claim) -> str:
    md = claim.source_metadata or {}
    for key in (
        "source_url",
        "original_claim_url",
        "cached_source_url",
        "cached_original_claim_url",
    ):
        val = md.get(key)
        if val:
            return str(val)
    return ""


def _effective_source_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return ""
    m = _ARCHIVE_EMBEDDED_URL_RE.search(url)
    if m:
        return unquote(m.group(1))
    return url


def _domain_suffix_match(domain: str, suffixes: Tuple[str, ...]) -> bool:
    return any(domain == s or domain.endswith("." + s) for s in suffixes)


def _domain_of(claim: Claim) -> str:
    raw_url = _source_url_of(claim)
    if not raw_url:
        return ""
    parsed = urlparse(_effective_source_url(raw_url))
    host = (parsed.netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_authority_score(domain: str) -> Optional[float]:
    d = str(domain or "").lower().strip()
    if not d:
        return None
    if d.endswith(".gov") or d.endswith(".edu"):
        return 0.9
    if _domain_suffix_match(d, _HIGH_AUTH_DOMAIN_SUFFIXES):
        return 0.86
    if _domain_suffix_match(d, _LOW_AUTH_DOMAIN_SUFFIXES):
        return 0.18
    return None


def _medium_of(claim: Claim) -> str:
    md = claim.source_metadata or {}
    medium = _normalize_medium(md.get("source_medium", ""))
    if medium:
        return medium
    return _infer_medium_from_reporting_source(md.get("reporting_source", ""))


def _is_opinion_medium(claim: Claim) -> bool:
    medium = _medium_of(claim)
    if medium in _OPINION_MEDIA:
        return True
    # Accept variants like "opinion_article" / "editorial_board".
    if any(tag in medium for tag in ("opinion", "editorial", "commentary", "column", "op-ed", "op_ed")):
        return True

    source_url = _source_url_of(claim)
    if source_url:
        path = (urlparse(_effective_source_url(source_url)).path or "").lower()
        if _OPINION_URL_RE.search(path):
            return True

    question = str((claim.source_metadata or {}).get("question") or "")
    if re.search(r"opinion|editorial|commentary", question, flags=re.IGNORECASE):
        return True
    return False


_ROLE_AUTHORITY_PRIOR = {
    "query": 0.55,
    "claim": 0.55,
    "headline": 0.55,
    "retrieved_evidence": 0.65,
    "evidence": 0.65,
    "body": 0.65,
    "snippet": 0.65,
    "chunk": 0.65,
}


def _source_authority_score(c: Claim) -> Optional[float]:
    md = c.source_metadata or {}
    for k in ("authority", "source_quality", "reliability"):
        v = md.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass

    medium = _medium_of(c)
    if medium in _OPINION_MEDIA:
        return None  # opinion ≠ low quality

    medium_score = _MEDIUM_AUTHORITY.get(medium)
    domain_score = _domain_authority_score(_domain_of(c))

    if medium_score is not None and domain_score is not None:
        return 0.5 * medium_score + 0.5 * domain_score
    if domain_score is not None:
        return domain_score
    if medium_score is not None:
        return medium_score

    role = str(c.role or md.get("role") or "").lower().strip()
    return _ROLE_AUTHORITY_PRIOR.get(role)


def _authority_gap(
    a: Claim,
    b: Claim,
    min_gap: float = _AUTHORITY_GAP_MIN,
) -> Optional[Tuple[Claim, Claim, float, float]]:
    """
    Return (low_quality_side, high_quality_side) if source metadata
    clearly marks one side as less authoritative. Returns None otherwise.

    Returns (low_side, high_side, low_score, high_score). Scores combine
    normalized medium/domain cues and optional numeric quality fields.
    """
    sa, sb = _source_authority_score(a), _source_authority_score(b)
    if sa is None or sb is None:
        return None
    if abs(sa - sb) < min_gap:
        return None
    if sa < sb:
        return (a, b, sa, sb)
    return (b, a, sb, sa)


# ---------------------------------------------------------------------------
# 3. Temporal — month-precision, claim_date aware
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(_MONTH_NAMES.keys())

_ISO_DATE_RE = re.compile(
    r"\b((?:19|20)\d{2})[-/](0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?\b"
)
_CHINESE_YM_RE = re.compile(r"((?:19|20)\d{2})年\s*(0?[1-9]|1[0-2])月")
_MONTH_NAME_YEAR_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+((?:19|20)\d{{2}})\b", flags=re.IGNORECASE
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _extract_year_month(*texts: Optional[str]) -> Optional[Tuple[int, int]]:
    for t in texts:
        if not t:
            continue
        s = str(t)
        m = _ISO_DATE_RE.search(s)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = _CHINESE_YM_RE.search(s)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = _MONTH_NAME_YEAR_RE.search(s)
        if m:
            return int(m.group(2)), _MONTH_NAMES[m.group(1).lower().rstrip(".")]
    return None


def _extract_year(*texts: Optional[str]) -> Optional[int]:
    for t in texts:
        if not t:
            continue
        m = _YEAR_RE.search(str(t))
        if m:
            return int(m.group(0))
    return None


def _temporal_signal(a: Claim, b: Claim) -> Optional[str]:
    """
    Return a human-readable description of the temporal gap, or None if
    no temporal conflict is detected. Checks in order of precision:
    (year, month) → year. Sources scanned in order:
        claim.time
        claim.source_metadata.claim_date
        claim.text
    """
    a_sources = (
        a.time,
        (a.source_metadata or {}).get("claim_date"),
        a.text,
    )
    b_sources = (
        b.time,
        (b.source_metadata or {}).get("claim_date"),
        b.text,
    )

    ym_a = _extract_year_month(*a_sources)
    ym_b = _extract_year_month(*b_sources)
    if ym_a and ym_b and ym_a != ym_b:
        return f"time markers differ: claim_a~{ym_a[0]}-{ym_a[1]:02d} vs claim_b~{ym_b[0]}-{ym_b[1]:02d}"

    year_a = _extract_year(*a_sources)
    year_b = _extract_year(*b_sources)
    if year_a is not None and year_b is not None and year_a != year_b:
        return f"time markers differ: claim_a~{year_a} vs claim_b~{year_b}"

    return None


# ---------------------------------------------------------------------------
# 4. Numerical clash — same subject, different headline stat
# ---------------------------------------------------------------------------

# Integers with thousand separators, decimals, or plain ints. Years (1900–2099)
# are filtered out below so the temporal rule stays the canonical source.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _extract_non_year_numbers(text: str) -> List[float]:
    if not text:
        return []
    out: List[float] = []
    for match in _NUMBER_RE.findall(text):
        try:
            val = float(match.replace(",", ""))
        except ValueError:
            continue
        # Treat year-like plain integers as handled by _temporal_signal.
        if val == int(val) and 1900 <= val <= 2099:
            continue
        out.append(val)
    return out


def _numbers_clash(a: Claim, b: Claim) -> Optional[Tuple[float, float]]:
    """
    Return the dominant numbers on each side if they differ by >10%
    relative magnitude. Uses the absolute-largest number from each
    claim (usually the headline statistic).
    """
    nums_a = _extract_non_year_numbers(a.text or "")
    nums_b = _extract_non_year_numbers(b.text or "")
    if not nums_a or not nums_b:
        return None
    na = max(nums_a, key=abs)
    nb = max(nums_b, key=abs)
    denom = max(abs(na), abs(nb))
    if denom == 0:
        return None
    if abs(na - nb) / denom < _NUMERICAL_REL_DIFF_MIN:
        return None
    return (na, nb)


# ---------------------------------------------------------------------------
# 5. Lexical contradiction rescue (agreement / neutral override)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
_NEGATION_RE = re.compile(
    r"\b(no|not|never|none|without|can't|cannot|isn't|aren't|don't|"
    r"doesn't|didn't|won't|neither|nor)\b|"
    r"(并非|不是|并不|没有|并未|未曾|无)",
    flags=re.IGNORECASE,
)
_REFUTATION_RE = re.compile(
    r"\b(false|fake|hoax|debunk(?:ed|s|ing)?|refut(?:e|ed|es|ing)|"
    r"deny|denied|denies|incorrect|inaccurate|misleading|fabricated|"
    r"fact[- ]?check(?:ed|ing)?)\b|"
    r"(辟谣|谣言|不实|虚假|造谣|驳斥|否认)",
    flags=re.IGNORECASE,
)
_CONTENT_STOPWORDS = frozenset({
    "about", "after", "before", "being", "between", "during", "their",
    "there", "these", "those", "where", "which", "would", "could",
    "should", "while", "through", "against", "because", "official",
    "report", "reported", "reports", "according", "claim", "claims",
    "people", "country", "state", "states", "government", "president",
    "minister", "years", "year", "months", "month", "today",
    "yesterday", "already", "still", "around",
})


def _content_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    toks = set()
    for t in _WORD_RE.findall(text.lower()):
        if len(t) < 4:
            continue
        if t in _CONTENT_STOPWORDS:
            continue
        toks.add(t)
        # Lightweight normalization to recover overlaps like
        # approve/approved, deny/denied without pulling in NLP deps.
        if t.endswith("e") and len(t) > 4:
            toks.add(t[:-1])
        if t.endswith("ed") and len(t) > 5:
            toks.add(t[:-2])
        if t.endswith("ing") and len(t) > 6:
            toks.add(t[:-3])
        if t.endswith("es") and len(t) > 5:
            toks.add(t[:-2])
        if t.endswith("s") and len(t) > 4:
            toks.add(t[:-1])
    return toks


def _agreement_or_neutral_override_reason(
    claim_a: Claim,
    claim_b: Claim,
    nli_label: Optional[str],
) -> Optional[str]:
    """
    Return a rationale fragment if lexical cues suggest hidden conflict
    despite an agreement/neutral fused signal.
    """
    if nli_label == "entailment":
        return None

    text_a = claim_a.text or ""
    text_b = claim_b.text or ""
    toks_a = _content_tokens(text_a)
    toks_b = _content_tokens(text_b)
    overlap = len(toks_a & toks_b)

    if overlap <= 0:
        return None

    has_refutation = bool(_REFUTATION_RE.search(text_a) or _REFUTATION_RE.search(text_b))
    if has_refutation and overlap >= _REFUTATION_OVERRIDE_MIN_OVERLAP:
        return (
            "agreement/neutral signal override: strong refutation markers "
            "on shared topic terms"
        )

    neg_a = bool(_NEGATION_RE.search(text_a))
    neg_b = bool(_NEGATION_RE.search(text_b))
    if neg_a != neg_b and overlap >= _NEGATION_OVERRIDE_MIN_OVERLAP:
        return (
            "agreement/neutral signal override: negation mismatch "
            "on shared topic terms"
        )

    return None


def _source_gap_override_reason(
    claim_a: Claim,
    claim_b: Claim,
    nli_label: Optional[str],
) -> Optional[str]:
    """
    In stance-only mode, a strong source-quality gap can be a useful hint
    that "agreement" is unreliable on out-of-domain snippets.
    """
    if nli_label is not None:
        return None

    gap = _authority_gap(claim_a, claim_b, min_gap=_SOURCE_GAP_OVERRIDE_MIN)
    if gap is None:
        return None

    low, high, low_s, high_s = gap
    if low.claim_id != claim_a.claim_id or high.claim_id != claim_b.claim_id:
        return None
    if low_s > 0.25 or high_s < 0.7:
        return None

    # Keep source-gap override for low-lexical-cue cases. If explicit
    # negation/refutation wording exists, lexical rules should decide.
    text_b = claim_b.text or ""
    if _NEGATION_RE.search(text_b) or _REFUTATION_RE.search(text_b):
        return None

    domain_b_score = _domain_authority_score(_domain_of(claim_b))
    if domain_b_score is None or domain_b_score < 0.8:
        return None

    return (
        "agreement/neutral signal override: strong source quality gap "
        f"claim_a~{low_s:.2f} vs claim_b~{high_s:.2f}"
    )


# ---------------------------------------------------------------------------
# 6. Subject disambiguation (same-name entity)
# ---------------------------------------------------------------------------

_PERSON_NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+)\b"
)


def _normalize_person_name(name: str) -> List[str]:
    cleaned = re.sub(r"[^A-Za-z\s]", " ", name)
    parts = [p for p in cleaned.lower().split() if p]
    return parts


def _same_name_entity_clash(text_a: str, text_b: str) -> bool:
    ma = _PERSON_NAME_RE.search(text_a or "")
    mb = _PERSON_NAME_RE.search(text_b or "")
    if ma is None or mb is None:
        return False

    pa = _normalize_person_name(ma.group(1))
    pb = _normalize_person_name(mb.group(1))
    if not pa or not pb:
        return False
    if pa == pb:
        return False

    # Example: "Michael Jordan" vs "Michael I Jordan".
    return (
        len(pa) >= 2
        and len(pb) >= 2
        and pa[0] == pb[0]
        and pa[-1] == pb[-1]
    )


def _subjects_look_different(a: Claim, b: Claim) -> bool:
    sa = (a.subject or "").strip().lower()
    sb = (b.subject or "").strip().lower()
    if sa and sb:
        return sa != sb
    return _same_name_entity_clash(a.text or "", b.text or "")


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def type_pair(
    stance: StancedPair,
    claim_a: Optional[Claim],
    claim_b: Optional[Claim],
) -> TypedPair:
    """Classify a single stanced pair into a conflict type."""
    rationale: List[str] = []

    if claim_a is None or claim_b is None:
        rationale.append("one of the claims could not be resolved from the InputRecord")
        return TypedPair(
            stance=stance,
            conflict_type="ambiguity",
            typing_confidence=0.1,
            resolution_policy=DEFAULT_POLICY_BY_TYPE["ambiguity"],
            rationale=rationale,
        )

    signal = stance.agreement_signal

    # Fast paths.
    if signal == "unrelated":
        rationale.append("stance filtered as unrelated -> noise")
        return _pack(stance, "noise", 0.8, rationale)

    if signal in ("agreement", "neutral"):
        source_gap_override = False
        override_reason = _agreement_or_neutral_override_reason(
            claim_a,
            claim_b,
            stance.nli_label,
        )
        if override_reason is None:
            override_reason = _source_gap_override_reason(
                claim_a,
                claim_b,
                stance.nli_label,
            )
            source_gap_override = override_reason is not None
        if override_reason is None:
            rationale.append(f"agreement_signal={signal} -> no conflict")
            return _pack(stance, "none", 0.8, rationale)
        rationale.append(
            f"agreement_signal={signal} but {override_reason}; continuing subtype checks"
        )
        if source_gap_override:
            base_type = "ambiguity"
            base_conf = 0.5
        else:
            base_type = "hard_contradiction" if signal == "agreement" else "ambiguity"
            base_conf = 0.45
    elif signal == "inconclusive":
        rationale.append("stance and NLI disagree; starting from 'ambiguity'")
        base_type = "ambiguity"
        base_conf = 0.5
    else:
        rationale.append("agreement_signal=conflict; looking for sub-type")
        base_type = "hard_contradiction"
        base_conf = 0.6

    # Rule pipeline — first match wins, ordered by how informative the cue is.

    # R1) Temporal (month precision; claim_date aware)
    temporal_note = _temporal_signal(claim_a, claim_b)
    if temporal_note is not None:
        rationale.append(f"{temporal_note} -> temporal_conflict")
        return _pack(stance, "temporal_conflict", min(0.9, base_conf + 0.2), rationale)

    # R2) Numerical clash -> ambiguity (different stats / methodology)
    clash = _numbers_clash(claim_a, claim_b)
    if clash is not None:
        na, nb = clash
        rationale.append(
            f"numerical clash on dominant stat: {na:g} vs {nb:g} -> ambiguity"
        )
        return _pack(stance, "ambiguity", min(0.8, base_conf + 0.15), rationale)

    # R3) Different subjects -> ambiguity (same-name entity)
    if _subjects_look_different(claim_a, claim_b):
        rationale.append(
            f"subjects differ: '{claim_a.subject}' vs '{claim_b.subject}' -> ambiguity"
        )
        return _pack(stance, "ambiguity", min(0.85, base_conf + 0.2), rationale)

    # R4a) Opinion medium (precise signal)
    if _is_opinion_medium(claim_a) or _is_opinion_medium(claim_b):
        rationale.append(
            f"op-ed / editorial medium detected "
            f"(a={_medium_of(claim_a)!r}, b={_medium_of(claim_b)!r}) -> opinion_conflict"
        )
        return _pack(stance, "opinion_conflict", min(0.75, base_conf + 0.1), rationale)

    # R5) Misinformation — authority gap
    gap = _authority_gap(claim_a, claim_b)
    if gap is not None:
        low, high, low_s, high_s = gap
        if signal in ("agreement", "neutral") and source_gap_override:
            rationale.append(
                f"source quality gap persists ({low_s:.2f} vs {high_s:.2f}); "
                "keeping ambiguity in stance-only override path"
            )
            return _pack(stance, "ambiguity", min(0.7, base_conf + 0.15), rationale)
        rationale.append(
            f"source quality gap detected: '{low.claim_id}' looks less "
            f"authoritative than '{high.claim_id}' ({low_s:.2f} vs {high_s:.2f}) "
            f"-> misinformation"
        )
        return _pack(stance, "misinformation", min(0.8, base_conf + 0.15), rationale)

    # R4b) Opinion text cues (broader but lower precision than medium metadata)
    if _has_opinion_marker(claim_a) or _has_opinion_marker(claim_b):
        rationale.append("opinion / reported-speech markers in claim text -> opinion_conflict")
        return _pack(stance, "opinion_conflict", min(0.7, base_conf + 0.1), rationale)

    # R6) Fallback
    rationale.append(f"no further cues matched; falling back to '{base_type}'")
    return _pack(stance, base_type, base_conf, rationale)


def _pack(
    stance: StancedPair, conflict_type: str, confidence: float, rationale: List[str]
) -> TypedPair:
    return TypedPair(
        stance=stance,
        conflict_type=conflict_type,
        typing_confidence=float(confidence),
        resolution_policy=DEFAULT_POLICY_BY_TYPE[conflict_type],
        rationale=list(rationale),
    )


def type_sample(
    stanced_sample,
    input_record: InputRecord,
    gold_verdict: Optional[str] = None,
) -> TypedSample:
    """Classify every pair in a stanced sample, using its InputRecord for lookup."""
    typed_pairs: List[TypedPair] = []
    for sp in stanced_sample.pair_results:
        a = input_record.get_claim(sp.claim_a_id)
        b = input_record.get_claim(sp.claim_b_id)
        typed_pairs.append(type_pair(sp, a, b))

    counts: Dict[str, int] = {}
    for tp in typed_pairs:
        counts[tp.conflict_type] = counts.get(tp.conflict_type, 0) + 1

    return TypedSample(
        sample_id=stanced_sample.sample_id,
        pair_results=typed_pairs,
        type_counts=counts,
        gold_verdict=gold_verdict,
    )
