"""Controlled registry of feature-engineering operations.

The LLM is never allowed to emit executable code. The only thing a provider may
return is a list of ``{"column": ..., "operation": ...}`` proposals, and every
proposal is checked against this registry plus the dataset profile before a
single transformation is applied. Anything unknown, incompatible, or aimed at
the target column is rejected and reported back to the caller instead of running.

Every operation is deterministic and needs the target column for nothing at all:
the only statistics they use (frequency maps, z-score moments, rare-category
sets) are captured on the training split by ``FeatureEngineer.fit``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Hard limits stop a chatty provider from turning into a wide, slow matrix.
MAX_OPERATIONS = 24
_MAX_NAME_LEN = 64
_MAX_PARAM_STR_LEN = 32

# Types that may never be engineered from: identifiers and constants carry no
# signal, and the target column is a label, never a feature.
UNOPERABLE_TYPES = ("id_like", "constant", "unknown")

NUMERIC_TYPES = ("numerical",)
CATEGORICAL_TYPES = ("categorical",)
TEXT_TYPES = ("text",)
DATETIME_TYPES = ("datetime",)


class OperationValidationError(ValueError):
    """Raised for malformed registry input, never for a rejected proposal."""


@dataclass(frozen=True)
class OperationSpec:
    """A single safe transformation the pipeline is allowed to perform."""

    name: str
    category: str                        # numeric | categorical | text | datetime
    applies_to: tuple[str, ...]          # profiler inferred types accepted
    description: str
    output_suffix: str = ""              # feature name becomes f"{column}_{suffix}"
    in_place: bool = False               # rewrites the source column instead of adding one
    pair_operation: bool = False         # needs a second column supplied in params
    params: dict[str, tuple] = field(default_factory=dict)  # name -> (type, min, max, default)

    def feature_name(self, column: str, other: Optional[str] = None) -> str:
        if self.in_place:
            return column
        if self.pair_operation and other:
            return f"{column}_x_{other}"
        return f"{column}_{self.output_suffix}" if self.output_suffix else column


_RARE_PARAMS = {"threshold": (float, 0.0, 0.5, 0.02), "group_label": (str, 0, 0, "Other")}
_PAIR_PARAMS = {"with_column": (str, 0, 0, "")}

NUMERIC_SPECS = (
    OperationSpec(
        name="log1p",
        category="numeric",
        applies_to=NUMERIC_TYPES,
        description="Signed log1p transform; use on skewed counts/money-like columns.",
        output_suffix="log1p",
    ),
    OperationSpec(
        name="sqrt",
        category="numeric",
        applies_to=NUMERIC_TYPES,
        description="Square root; mild skew reduction for non-negative measures.",
        output_suffix="sqrt",
    ),
    OperationSpec(
        name="square",
        category="numeric",
        applies_to=NUMERIC_TYPES,
        description="Square of the value; exposes curvature a linear model cannot fit.",
        output_suffix="square",
    ),
    OperationSpec(
        name="absolute",
        category="numeric",
        applies_to=NUMERIC_TYPES,
        description="Absolute magnitude of the value.",
        output_suffix="abs",
    ),
    OperationSpec(
        name="zscore",
        category="numeric",
        applies_to=NUMERIC_TYPES,
        description="Z-score using training-split mean/std (never computed on held-out rows).",
        output_suffix="zscore",
    ),
    OperationSpec(
        name="interaction",
        category="numeric",
        applies_to=NUMERIC_TYPES,
        description="Product of two numeric columns; pass the partner in params.with_column.",
        pair_operation=True,
        params=_PAIR_PARAMS,
    ),
)

CATEGORICAL_SPECS = (
    OperationSpec(
        name="frequency_encoding",
        category="categorical",
        applies_to=CATEGORICAL_TYPES,
        description="Replace each category by its training-split relative frequency.",
        output_suffix="freq",
    ),
    OperationSpec(
        name="rare_category_grouping",
        category="categorical",
        applies_to=CATEGORICAL_TYPES,
        description="Bucket infrequent categories into one label before one-hot encoding.",
        in_place=True,
        params=_RARE_PARAMS,
    ),
)

TEXT_SPECS = (
    OperationSpec(
        name="char_count",
        category="text",
        applies_to=TEXT_TYPES,
        description="Number of characters in the text.",
        output_suffix="char_count",
    ),
    OperationSpec(
        name="word_count",
        category="text",
        applies_to=TEXT_TYPES,
        description="Number of whitespace separated words in the text.",
        output_suffix="word_count",
    ),
    OperationSpec(
        name="text_length",
        category="text",
        applies_to=TEXT_TYPES,
        description="Number of non-whitespace characters in the text.",
        output_suffix="text_length",
    ),
)

DATETIME_SPECS = (
    OperationSpec(
        name="year",
        category="datetime",
        applies_to=DATETIME_TYPES,
        description="Calendar year parsed from the timestamp.",
        output_suffix="year",
    ),
    OperationSpec(
        name="month",
        category="datetime",
        applies_to=DATETIME_TYPES,
        description="Calendar month (1-12) parsed from the timestamp.",
        output_suffix="month",
    ),
    OperationSpec(
        name="day",
        category="datetime",
        applies_to=DATETIME_TYPES,
        description="Day of month (1-31) parsed from the timestamp.",
        output_suffix="day",
    ),
    OperationSpec(
        name="dayofweek",
        category="datetime",
        applies_to=DATETIME_TYPES,
        description="Day of week (0=Monday) parsed from the timestamp.",
        output_suffix="dayofweek",
    ),
)

DATETIME_OPERATION_NAMES = tuple(spec.name for spec in DATETIME_SPECS)

OPERATION_REGISTRY: dict[str, OperationSpec] = {
    spec.name: spec
    for spec in (*NUMERIC_SPECS, *CATEGORICAL_SPECS, *TEXT_SPECS, *DATETIME_SPECS)
}


# Spellings models reach for that map onto a registered operation. Keys are the
# normalized (lower_snake) form of the model's wording.
OPERATION_ALIASES: dict[str, str] = {
    "log": "log1p",
    "log_transform": "log1p",
    "log1p_transform": "log1p",
    "logarithmic": "log1p",
    "signed_log1p": "log1p",
    "sqrt_transform": "sqrt",
    "square_root": "sqrt",
    "squared": "square",
    "power2": "square",
    "x_squared": "square",
    "abs": "absolute",
    "abs_value": "absolute",
    "absolute_value": "absolute",
    "z_score": "zscore",
    "standardize": "zscore",
    "standard_scale": "zscore",
    "standardisation": "zscore",
    "standardization": "zscore",
    "multiply": "interaction",
    "product": "interaction",
    "pairwise_interaction": "interaction",
    "feature_interaction": "interaction",
    "frequency": "frequency_encoding",
    "frequency_encode": "frequency_encoding",
    "freq_encoding": "frequency_encoding",
    "rate_encoding": "frequency_encoding",
    "target_frequency_encoding": "frequency_encoding",
    "group_rare_categories": "rare_category_grouping",
    "rare_category_bucketing": "rare_category_grouping",
    "other_category_grouping": "rare_category_grouping",
    "hash_grouping": "rare_category_grouping",
    "length": "char_count",
    "string_length": "char_count",
    "character_count": "char_count",
    "num_characters": "char_count",
    "n_characters": "char_count",
    "text_char_count": "char_count",
    "word_counter": "word_count",
    "words_count": "word_count",
    "num_words": "word_count",
    "n_words": "word_count",
    "token_count": "word_count",
    "text_word_count": "word_count",
    "dt_year": "year",
    "dt_month": "month",
    "dt_day": "day",
    "dt_dayofweek": "dayofweek",
    "day_of_week": "dayofweek",
    "weekday": "dayofweek",
    "extract_year": "year",
    "extract_month": "month",
    "extract_day": "day",
    "extract_dayofweek": "dayofweek",
}


def normalize_operation_name(raw: Any) -> Optional[str]:
    """Map a model-supplied operation string onto a registry name (or None)."""
    if not isinstance(raw, str):
        return None
    token = raw.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    token = token.strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    if not token or len(token) > _MAX_NAME_LEN:
        return None
    if token in OPERATION_REGISTRY:
        return token
    for prefix in ("apply_", "add_", "create_", "make_", "compute_", "generate_"):
        if token.startswith(prefix) and token[len(prefix):] in OPERATION_REGISTRY:
            return token[len(prefix):]
    alias = OPERATION_ALIASES.get(token)
    if alias and alias in OPERATION_REGISTRY:
        return alias
    for suffix in ("_transform", "_encoding", "_feature"):
        if token.endswith(suffix):
            alias = OPERATION_ALIASES.get(token[:-len(suffix)])
            if alias and alias in OPERATION_REGISTRY:
                return alias
            if token[:-len(suffix)] in OPERATION_REGISTRY:
                return token[:-len(suffix)]
    return None



def operation_catalog_descriptions() -> dict[str, str]:
    """Registry-wide catalogue (name -> "[category] description") for prompts/UI."""
    return {name: f"[{spec.category}] {spec.description}"
            for name, spec in sorted(OPERATION_REGISTRY.items())}


def allowed_operations_for_profile(profile: dict) -> list[str]:
    """Registry operations legal for one profiled column, in registry order."""
    inferred = str(profile.get("inferred_type") or "")
    if inferred in UNOPERABLE_TYPES:
        return []
    return [name for name, spec in OPERATION_REGISTRY.items() if inferred in spec.applies_to]


def build_operation_catalog(
    column_profiles: list[dict] | dict[str, dict] | None,
    target_column: Optional[str] = None,
) -> dict[str, list[str]]:
    """Map every *candidate input column* to its allowed operation names.

    The target column is deliberately absent: proposals aimed at the label are a
    leakage risk and get rejected again during validation.
    """
    if column_profiles is None:
        return {}
    profiles = (column_profiles.values() if isinstance(column_profiles, dict)
                else column_profiles)
    catalog: dict[str, list[str]] = {}
    for profile in profiles or []:
        if not isinstance(profile, dict):
            continue
        name = profile.get("name")
        if not isinstance(name, str) or not name or name == target_column:
            continue
        allowed = allowed_operations_for_profile(profile)
        if allowed:
            catalog[name] = allowed
    return catalog


def _normalize_column_name(raw: Any) -> Optional[str]:
    """Accept any existing column name verbatim (names are never rewritten)."""
    if not isinstance(raw, str):
        return None
    name = raw.strip()
    return name if name and len(name) <= _MAX_NAME_LEN else None


def _validate_params(spec: OperationSpec, raw: Any) -> tuple[dict, list[str]]:
    """Coerce/bound params against the spec; returns (params, problems)."""
    problems: list[str] = []
    params: dict[str, Any] = {}
    if raw in (None, {}):
        return {p: d for p, (_, _, _, d) in spec.params.items()}, problems
    if not isinstance(raw, dict):
        return params, [f"params must be an object, got {type(raw).__name__}"]

    for key, value in raw.items():
        pname = str(key).strip()
        if pname not in spec.params:
            problems.append(f"unsupported param '{pname}'")
            continue
        expected, lo, hi, default = spec.params[pname]
        try:
            if expected is str:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("expected a non-empty string")
                params[pname] = value.strip()[:_MAX_PARAM_STR_LEN]
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("expected a number")
                number = float(value)
                if number != number:
                    raise ValueError("expected a finite number")
                if not (lo <= number <= hi):
                    problems.append(f"param '{pname}'={number} is outside [{lo}, {hi}]")
                    params[pname] = default
                else:
                    params[pname] = number
        except (TypeError, ValueError) as exc:
            problems.append(f"param '{pname}' is invalid ({exc})")
            params[pname] = default

    for pname, pdef in spec.params.items():
        params.setdefault(pname, pdef[3])
    if spec.pair_operation:
        partner = params.get("with_column")
        if not isinstance(partner, str) or not partner.strip():
            problems.append("this operation needs params.with_column")
        else:
            params["with_column"] = partner.strip()[:_MAX_NAME_LEN]
    return params, problems



def validate_operations(
    proposals: Any,
    column_profiles: "list[dict] | dict[str, dict] | None",
    target_column: Optional[str] = None,
    *,
    max_operations: int = MAX_OPERATIONS,
) -> tuple[list[dict], list[dict]]:
    """Split model proposals into ``(accepted, rejected)``.

    Args:
        proposals: raw ``suggested_operations`` payload - a list of objects.
        column_profiles: profiler ``column_profiles`` list (or name -> profile map).
        target_column: the label; anything referencing it is rejected outright.
        max_operations: cap on accepted operations so one chatty answer cannot
            explode the feature space.

    Accepted entries are normalized::

        {"column": str, "operation": str, "params": dict, "reason": str | None,
         "category": str, "features": [str]}

    Rejected entries carry ``column``, ``operation``, ``raw_operation`` and a
    human-readable ``reason`` so the API can say exactly what was discarded and
    why. Nothing here raises for a bad proposal - rejection is the answer.
    """
    if isinstance(column_profiles, dict):
        profiles = {k: v for k, v in column_profiles.items() if isinstance(v, dict)}
    else:
        profiles = {p["name"]: p for p in (column_profiles or [])
                    if isinstance(p, dict) and isinstance(p.get("name"), str)}
    catalog = build_operation_catalog(profiles, target_column)

    if proposals is None:
        return [], []
    if not isinstance(proposals, list):
        return [], [{"column": None, "operation": None,
                     "reason": f"expected a list of operations, got {type(proposals).__name__}"}]

    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[tuple] = set()

    for entry in proposals:
        if not isinstance(entry, dict):
            rejected.append({"column": None, "operation": None,
                             "reason": f"expected an object, got {type(entry).__name__}"})
            continue

        column = _normalize_column_name(entry.get("column"))
        raw_operation = entry.get("operation")
        raw_label = raw_operation if isinstance(raw_operation, str) else repr(raw_operation)
        operation = normalize_operation_name(raw_operation)
        note = _normalize_column_name(entry.get("reason"))
        ctx = {"column": column, "operation": operation, "raw_operation": raw_label,
               "suggested_reason": note}

        if operation is None:
            rejected.append({**ctx, "reason":
                             f"operation '{raw_label}' is not in the allowed operation registry"})
            continue
        if column is None:
            rejected.append({**ctx, "reason": "missing or invalid 'column'"})
            continue
        if target_column and column == target_column:
            rejected.append({**ctx, "reason":
                             "the target column is a label and must never be engineered"})
            continue
        if column not in catalog:
            if column not in profiles:
                rejected.append({**ctx, "reason": f"column '{column}' does not exist in the dataset"})
            else:
                rejected.append({**ctx, "reason":
                                 f"column '{column}' (type "
                                 f"{profiles[column].get('inferred_type')}) supports no operations"})
            continue

        spec = OPERATION_REGISTRY[operation]
        if operation not in catalog[column]:
            rejected.append({**ctx, "reason":
                             f"'{operation}' is not allowed on '{column}' (type "
                             f"{profiles[column].get('inferred_type')}); allowed: "
                             f"{', '.join(catalog[column])}"})
            continue

        params, param_problems = _validate_params(spec, entry.get("params"))
        if param_problems:
            rejected.append({**ctx, "reason": "; ".join(param_problems)})
            continue

        partner = params.get("with_column") if spec.pair_operation else None
        if spec.pair_operation:
            partner_profile = profiles.get(partner) if partner else None
            if not partner:
                rejected.append({**ctx, "reason": "params.with_column is required"})
                continue
            if partner == column:
                rejected.append({**ctx, "reason": "params.with_column must differ from 'column'"})
                continue
            if target_column and partner == target_column:
                rejected.append({**ctx, "reason": "params.with_column must not be the target column"})
                continue
            if partner_profile is None:
                rejected.append({**ctx, "reason": f"params.with_column '{partner}' is not a dataset column"})
                continue
            if partner not in catalog:
                rejected.append({**ctx, "reason":
                                 f"params.with_column '{partner}' is not a numeric input column"})
                continue

        key = (column, operation, tuple(sorted((k, str(v)) for k, v in params.items())))
        if key in seen:
            rejected.append({**ctx, "reason": "duplicate operation"})
            continue
        seen.add(key)

        if len(accepted) >= max_operations:
            rejected.append({**ctx, "reason": f"operation budget of {max_operations} exhausted"})
            continue

        accepted.append({
            "column": column,
            "operation": operation,
            "params": params,
            "reason": note,
            "category": spec.category,
            "features": [spec.feature_name(column, partner)],
        })

    return accepted, rejected

