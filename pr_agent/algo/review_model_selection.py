"""Typed, request-scoped model selections for the ``/review`` command."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from pr_agent.algo.utils import ReasoningEffort

_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# Each selector adds a provider attempt on the failure path, so keep caller-controlled
# chains small even when every model identity is operator-allowlisted.
MAX_MODEL_SELECTIONS = 4
_active_review_model_selection: ContextVar["ReviewModelSelection | None"] = ContextVar(
    "pr_agent_active_review_model_selection", default=None
)


@dataclass(frozen=True)
class ReviewModelSelection:
    """One operator-allowlisted model and effort pair for a review attempt."""

    alias: str
    model: str
    reasoning_effort: str


class ReviewModelSelectionError(ValueError):
    """An actionable error caused by an invalid command model selector."""


def get_active_review_model_selection() -> ReviewModelSelection | None:
    """Return the selection active for the current fallback attempt, if any."""
    return _active_review_model_selection.get()


@contextmanager
def use_review_model_selection(selection: ReviewModelSelection) -> Iterator[None]:
    """Make ``selection`` visible to every AI call in one review attempt."""
    token = _active_review_model_selection.set(selection)
    try:
        yield
    finally:
        _active_review_model_selection.reset(token)


def _is_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _get_aliases(settings) -> dict[str, str]:
    raw_aliases = settings.get("PR_REVIEWER.COMMAND_MODEL_ALIASES", {}) or {}
    if not isinstance(raw_aliases, Mapping):
        raise ReviewModelSelectionError(
            "The operator configuration `pr_reviewer.command_model_aliases` must be a TOML mapping."
        )

    aliases = {}
    for raw_alias, raw_model in raw_aliases.items():
        alias = str(raw_alias).strip().lower()
        if not _ALIAS_RE.fullmatch(alias):
            raise ReviewModelSelectionError(
                f"The configured model alias `{raw_alias}` is invalid; use letters, numbers, `.`, `_`, or `-`."
            )
        if not isinstance(raw_model, str) or not raw_model.strip():
            raise ReviewModelSelectionError(
                f"The configured model alias `{raw_alias}` must map to a non-empty model identifier."
            )
        aliases[alias] = raw_model.strip()
    return aliases


def parse_review_model_selections(
    args: Sequence[str], settings
) -> tuple[tuple[ReviewModelSelection, ...], list[str]]:
    """Parse ordered ``alias+effort`` tokens and return the untouched remaining args.

    Tokens without ``+`` retain their historical meaning. This keeps ``/review`` and
    existing flags byte-for-byte compatible when no selector syntax is present.
    """
    selector_tokens = [arg for arg in args if "+" in arg]
    if not selector_tokens:
        return (), list(args)

    if not _is_enabled(settings.get("PR_REVIEWER.ENABLE_COMMAND_MODEL_OVERRIDES", False)):
        raise ReviewModelSelectionError(
            "Per-command model overrides are disabled. Ask an operator to enable "
            "`pr_reviewer.enable_command_model_overrides` in trusted global configuration."
        )

    aliases = _get_aliases(settings)
    if not aliases:
        raise ReviewModelSelectionError(
            "No command model aliases are configured. Ask an operator to set "
            "`pr_reviewer.command_model_aliases` in trusted global configuration."
        )

    selections = []
    remaining_args = []
    valid_efforts = [effort.value for effort in reversed(list(ReasoningEffort))]
    for arg in args:
        if "+" not in arg:
            remaining_args.append(arg)
            continue
        if arg.count("+") != 1:
            raise ReviewModelSelectionError(
                f"Malformed model selector `{arg}`. Use exactly `alias+effort`, for example `opus+high`."
            )
        raw_alias, raw_effort = arg.split("+", 1)
        alias = raw_alias.strip().lower()
        effort = raw_effort.strip().lower()
        if not alias or not effort:
            raise ReviewModelSelectionError(
                f"Malformed model selector `{arg}`. Use `alias+effort`, for example `opus+high`."
            )
        if "/" in alias or ":" in alias:
            raise ReviewModelSelectionError(
                f"Raw model identifier `{raw_alias}` is not allowed. Use an operator-configured alias instead."
            )
        if alias not in aliases:
            available = ", ".join(sorted(aliases))
            raise ReviewModelSelectionError(
                f"Unknown model alias `{raw_alias}`. Available aliases: {available}."
            )
        try:
            effort = ReasoningEffort(effort).value
        except ValueError as e:
            raise ReviewModelSelectionError(
                f"Unsupported reasoning effort `{raw_effort}`. Choose one of: {', '.join(valid_efforts)}."
            ) from e
        selection = ReviewModelSelection(
            alias=alias, model=aliases[alias], reasoning_effort=effort
        )
        if selection in selections:
            raise ReviewModelSelectionError(
                f"Duplicate model selector `{arg}`. List each `alias+effort` at most once."
            )
        if len(selections) >= MAX_MODEL_SELECTIONS:
            raise ReviewModelSelectionError(
                f"Too many model selectors (limit {MAX_MODEL_SELECTIONS}). Shorten the fallback chain."
            )
        selections.append(selection)

    return tuple(selections), remaining_args
