"""Safety: the Thai-aware tier-0 filter, the layered gate and the moderation audit (§7).

- :class:`KeywordRegexFilter` (``TextFilter``): token/substring/regex lists under
  ``config/filters/base`` plus private, per-platform and per-character overlays.
- :class:`LayeredSafetyGate` (``SafetyGate``): input, output (with ``prev_tail``), tool/memory/
  game arguments and display names; auto-strict; the tier-1 ``Classifier`` hook (M2).
- :class:`ModerationAudit`: PII-masked ``moderation_log`` rows through an injected sink.

Importing this package does not load pythainlp; the filter warms it when built.
"""

from aivtube.safety.audit import (
    ModerationAudit,
    ModerationRecord,
    ModerationSink,
    SupportsLogModeration,
    ops_sink,
    sha256_text,
)
from aivtube.safety.factory import build_keyword_filter, build_safety_gate
from aivtube.safety.gate import ANON_NAME, LayeredSafetyGate, check_name
from aivtube.safety.keyword import TIER, KeywordRegexFilter, category_verdict
from aivtube.safety.lists import CATEGORIES, FilterListError
from aivtube.safety.normalize import MatchForms, match_forms
from aivtube.safety.pii import PiiSpan, find_pii, mask_pii

__all__ = [
    "ANON_NAME",
    "CATEGORIES",
    "TIER",
    "FilterListError",
    "KeywordRegexFilter",
    "LayeredSafetyGate",
    "MatchForms",
    "ModerationAudit",
    "ModerationRecord",
    "ModerationSink",
    "PiiSpan",
    "SupportsLogModeration",
    "build_keyword_filter",
    "build_safety_gate",
    "category_verdict",
    "check_name",
    "find_pii",
    "mask_pii",
    "match_forms",
    "ops_sink",
    "sha256_text",
]
