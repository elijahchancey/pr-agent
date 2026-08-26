import asyncio
import copy
from contextlib import suppress
from unittest.mock import MagicMock

import pytest

import pr_agent.agent.pr_agent as pr_agent_module
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.cli_args import CliArgs
from pr_agent.algo.pr_processing import retry_with_fallback_models
from pr_agent.algo.review_model_selection import (
    ReviewModelSelection, ReviewModelSelectionError,
    get_active_review_model_selection, parse_review_model_selections,
    use_review_model_selection)
from pr_agent.algo.run_details import get_run_details, init_run_details
from pr_agent.config_loader import get_settings
from pr_agent.servers.help import HelpMessage
from tests.unittest._settings_helpers import (restore_settings,
                                              snapshot_settings)


class _Settings:
    def __init__(self, *, enabled=True, aliases=None):
        self.values = {
            "PR_REVIEWER.ENABLE_COMMAND_MODEL_OVERRIDES": enabled,
            "PR_REVIEWER.COMMAND_MODEL_ALIASES": aliases if aliases is not None else {
                "fable": "anthropic/claude-fable-5",
                "opus": "anthropic/claude-opus-5",
                "terra": "gpt-5.6-terra",
            },
        }

    def get(self, key, default=None):
        return self.values.get(key, default)


def _snapshot_sections(*names):
    settings_dict = get_settings().as_dict()
    return {name: copy.deepcopy(settings_dict.get(name)) for name in names}


def _restore_sections(snapshot):
    settings = get_settings()
    for name, value in snapshot.items():
        with suppress(KeyError):
            settings.unset(name, force=True)
        if value is not None:
            settings.set(name, value, merge=False)


def _replace_section_values(section_name, **values):
    settings = get_settings()
    section = copy.deepcopy(settings.as_dict().get(section_name, {}))
    for key, value in values.items():
        for stored_key in list(section):
            if stored_key.lower() == key.lower():
                section.pop(stored_key)
        section[key.lower()] = value
    with suppress(KeyError):
        settings.unset(section_name, force=True)
    settings.set(section_name, section, merge=False)


def test_review_without_selectors_does_not_read_override_configuration():
    class _UnusableSettings:
        def get(self, *_args, **_kwargs):
            raise AssertionError("ordinary /review must not inspect override configuration")

    selections, remaining_args = parse_review_model_selections(["-i", "legacy-arg"], _UnusableSettings())

    assert selections == ()
    assert remaining_args == ["-i", "legacy-arg"]


@pytest.mark.parametrize("settings", [_Settings(enabled=False), _Settings()])
def test_plus_tokens_that_are_not_selectors_keep_historical_meaning(settings):
    args = ["please", "check", "the", "C++", "parts", "a+b", "foo+bar"]

    selections, remaining_args = parse_review_model_selections(args, settings)

    assert selections == ()
    assert remaining_args == args


def test_malformed_alias_configuration_does_not_fail_reviews_without_selectors():
    settings = _Settings(aliases={"BAD ALIAS!": "anthropic/claude-opus-5"})

    selections, remaining_args = parse_review_model_selections(["C++", "a+b"], settings)

    assert selections == ()
    assert remaining_args == ["C++", "a+b"]
    with pytest.raises(ReviewModelSelectionError, match="invalid"):
        parse_review_model_selections(["opus+high"], settings)


def test_disabled_feature_ignores_effort_typos_instead_of_failing_the_review():
    selections, remaining_args = parse_review_model_selections(
        ["opus+extreme"], _Settings(enabled=False)
    )

    assert selections == ()
    assert remaining_args == ["opus+extreme"]


def test_review_help_documents_ordered_alias_effort_selectors():
    assert "/review [alias+effort ...]" in HelpMessage.get_general_commands_text()
    review_help = HelpMessage.get_review_usage_guide()
    assert "/review fable+high opus+high" in review_help
    assert "fallbacks for this review only" in review_help


def test_one_valid_selector_resolves_the_allowlisted_model_and_effort():
    selections, remaining_args = parse_review_model_selections(["opus+xhigh"], _Settings())

    assert selections == (
        ReviewModelSelection(
            alias="opus",
            model="anthropic/claude-opus-5",
            reasoning_effort="xhigh",
        ),
    )
    assert remaining_args == []


def test_two_selectors_preserve_order_and_other_review_arguments():
    selections, remaining_args = parse_review_model_selections(
        ["fable+high", "-i", "opus+low"], _Settings()
    )

    assert [(selection.alias, selection.reasoning_effort) for selection in selections] == [
        ("fable", "high"),
        ("opus", "low"),
    ]
    assert remaining_args == ["-i"]


def test_four_distinct_selectors_are_allowed():
    selections, remaining_args = parse_review_model_selections(
        ["opus+minimal", "opus+low", "opus+high", "opus+xhigh"], _Settings()
    )

    assert [selection.reasoning_effort for selection in selections] == [
        "minimal",
        "low",
        "high",
        "xhigh",
    ]
    assert remaining_args == []


@pytest.mark.parametrize(
    ("args", "settings", "message"),
    [
        (["opus+high"], _Settings(enabled=False), "overrides are disabled"),
        (["opus+high"], _Settings(aliases={}), "No command model aliases"),
        (["unknown+high"], _Settings(), "Unknown model alias"),
        (["opus++high"], _Settings(), "Malformed model selector"),
        (["opus+"], _Settings(), "Malformed model selector"),
        (["opus+extreme"], _Settings(), "Unsupported reasoning effort"),
        (["anthropic/claude-opus-5+high"], _Settings(), "Raw model identifier"),
        (["opus+high", "opus+high"], _Settings(), "Duplicate model selector"),
        (
            ["opus+none", "opus+minimal", "opus+low", "opus+medium", "opus+high"],
            _Settings(),
            "Too many model selectors",
        ),
    ],
)
def test_invalid_selectors_raise_actionable_errors(args, settings, message):
    with pytest.raises(ReviewModelSelectionError, match=message):
        parse_review_model_selections(args, settings)


def test_selector_fallback_uses_its_own_effort_and_does_not_leak():
    tracked_keys = (
        "config.model",
        "config.fallback_models",
        "openai.deployment_id",
        "openai.fallback_deployments",
    )
    snapshot = snapshot_settings(tracked_keys)
    settings = get_settings()
    try:
        settings.set("config.model", "configured-primary")
        settings.set("config.fallback_models", [])
        settings.set("openai.deployment_id", None)
        settings.set("openai.fallback_deployments", ["configured-fallback-deployment"])
        selections = (
            ReviewModelSelection("fable", "anthropic/claude-fable-5", "high"),
            ReviewModelSelection("opus", "anthropic/claude-opus-5", "low"),
        )
        attempts = []

        async def selected_call(model):
            active = get_active_review_model_selection()
            attempts.append((model, active.reasoning_effort if active else None))
            if model == "anthropic/claude-fable-5":
                raise RuntimeError("primary failed")
            return "fallback result"

        init_run_details()
        result = asyncio.run(retry_with_fallback_models(selected_call, model_selections=selections))

        assert result == "fallback result"
        assert attempts == [
            ("anthropic/claude-fable-5", "high"),
            ("anthropic/claude-opus-5", "low"),
        ]
        assert get_active_review_model_selection() is None
        assert settings.config.model == "configured-primary"
        assert settings.config.fallback_models == []
        details = get_run_details()
        assert details.model_used == "anthropic/claude-opus-5"
        assert details.reasoning_effort == "low"
        assert details.fallback_used is True

        ordinary_attempts = []

        async def ordinary_call(model):
            ordinary_attempts.append((model, get_active_review_model_selection()))
            return "ordinary result"

        assert asyncio.run(retry_with_fallback_models(ordinary_call)) == "ordinary result"
        assert ordinary_attempts == [("configured-primary", None)]
    finally:
        restore_settings(snapshot)


def _make_bare_handler(monkeypatch, captured):
    handler = LiteLLMAIHandler.__new__(LiteLLMAIHandler)
    handler.azure = False
    handler.api_base = None
    handler.repetition_penalty = None
    handler.claude_extended_thinking_models = []
    handler.no_support_temperature_models = []
    handler.support_reasoning_models = []
    handler.user_message_only_models = []
    handler._aws_imds_mode = False
    handler._aws_imds_fell_back = False
    handler._aws_static_creds = None
    handler._aws_bedrock_lock = None

    async def fake_get_completion(**kwargs):
        captured.update(kwargs)
        response = MagicMock()
        response.usage = None
        response.dict.return_value = {
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}]
        }
        return "response", "stop", response

    monkeypatch.setattr(handler, "_get_completion", fake_get_completion)
    return handler


def test_litellm_applies_command_effort_to_an_allowlisted_model_outside_builtin_lists(monkeypatch):
    captured = {}
    handler = _make_bare_handler(monkeypatch, captured)
    selection = ReviewModelSelection("opus", "anthropic/claude-opus-5", "xhigh")

    with use_review_model_selection(selection):
        response, finish_reason = asyncio.run(
            handler.chat_completion(model=selection.model, system="system", user="user")
        )

    assert (response, finish_reason) == ("response", "stop")
    assert captured["model"] == "anthropic/claude-opus-5"
    assert captured["reasoning_effort"] == "xhigh"
    # Without this, litellm rejects the effort client-side for models missing
    # from its capability map (drop_params defaults to false).
    assert captured["allowed_openai_params"] == ["reasoning_effort"]
    # Anthropic rejects a pinned temperature while extended thinking is enabled.
    assert "temperature" not in captured


def test_litellm_command_effort_none_keeps_temperature_for_claude(monkeypatch):
    captured = {}
    handler = _make_bare_handler(monkeypatch, captured)
    selection = ReviewModelSelection("opus", "anthropic/claude-opus-5", "none")

    with use_review_model_selection(selection):
        asyncio.run(handler.chat_completion(model=selection.model, system="system", user="user"))

    assert captured["reasoning_effort"] == "none"
    assert captured["temperature"] == 0.2


def test_litellm_command_effort_keeps_temperature_for_non_anthropic_models(monkeypatch):
    captured = {}
    handler = _make_bare_handler(monkeypatch, captured)
    selection = ReviewModelSelection("terra", "some-provider/new-model", "high")

    with use_review_model_selection(selection):
        asyncio.run(handler.chat_completion(model=selection.model, system="system", user="user"))

    assert captured["reasoning_effort"] == "high"
    assert captured["allowed_openai_params"] == ["reasoning_effort"]
    assert captured["temperature"] == 0.2


@pytest.mark.parametrize(
    "arg",
    [
        "--pr_reviewer.enable_command_model_overrides=true",
        "--pr_reviewer.command_model_aliases.opus=anthropic/claude-opus-5",
    ],
)
def test_comment_configuration_cannot_change_operator_override_controls(arg):
    is_valid, forbidden_arg = CliArgs.validate_user_args([arg])

    assert is_valid is False
    assert forbidden_arg


@pytest.mark.asyncio
async def test_review_without_selectors_uses_the_existing_constructor_path(monkeypatch):
    snapshot = _snapshot_sections("CONFIG")
    reviewer_calls = []

    class _ExistingReviewer:
        def __init__(self, pr_url, ai_handler, args):
            reviewer_calls.append((pr_url, ai_handler, args))

        async def run(self):
            pass

    try:
        _replace_section_values("CONFIG", RESPONSE_LANGUAGE="en-us")
        monkeypatch.setattr(pr_agent_module, "apply_repo_settings", lambda _pr_url: None)
        monkeypatch.setitem(pr_agent_module.command2class, "review", _ExistingReviewer)

        handled = await pr_agent_module.PRAgent(ai_handler="fake-ai")._handle_request(
            "https://example/pr/1", "/review -i"
        )

        assert handled is True
        assert reviewer_calls == [("https://example/pr/1", "fake-ai", ["-i"])]
    finally:
        _restore_sections(snapshot)


@pytest.mark.asyncio
async def test_plain_review_comment_mentioning_cpp_runs_with_feature_disabled(monkeypatch):
    snapshot = _snapshot_sections("CONFIG")
    reviewer_calls = []

    class _ExistingReviewer:
        def __init__(self, pr_url, ai_handler, args):
            reviewer_calls.append((pr_url, ai_handler, args))

        async def run(self):
            pass

    try:
        _replace_section_values("CONFIG", RESPONSE_LANGUAGE="en-us")
        monkeypatch.setattr(pr_agent_module, "apply_repo_settings", lambda _pr_url: None)
        monkeypatch.setitem(pr_agent_module.command2class, "review", _ExistingReviewer)

        handled = await pr_agent_module.PRAgent(ai_handler="fake-ai")._handle_request(
            "https://example/pr/1", "/review please check the C++ parts"
        )

        assert handled is True
        assert reviewer_calls == [
            ("https://example/pr/1", "fake-ai", ["please", "check", "the", "C++", "parts"])
        ]
    finally:
        _restore_sections(snapshot)


@pytest.mark.asyncio
async def test_valid_selector_keeps_incremental_and_config_arguments_working(monkeypatch):
    snapshot = _snapshot_sections("CONFIG", "PR_REVIEWER")
    settings = get_settings()
    reviewer_calls = []

    class _Reviewer:
        def __init__(self, pr_url, args, ai_handler, model_selections):
            reviewer_calls.append((pr_url, args, ai_handler, model_selections))

        async def run(self):
            pass

    try:
        _replace_section_values("CONFIG", RESPONSE_LANGUAGE="en-us")
        _replace_section_values(
            "PR_REVIEWER",
            ENABLE_COMMAND_MODEL_OVERRIDES=True,
            COMMAND_MODEL_ALIASES={"terra": "gpt-5.6-terra"},
            EXTRA_INSTRUCTIONS="before",
        )
        monkeypatch.setattr(pr_agent_module, "apply_repo_settings", lambda _pr_url: None)
        monkeypatch.setitem(pr_agent_module.command2class, "review", _Reviewer)

        handled = await pr_agent_module.PRAgent(ai_handler="fake-ai")._handle_request(
            "https://example/pr/1",
            "/review terra+low -i --pr_reviewer.extra_instructions=focused",
        )

        assert handled is True
        assert settings.pr_reviewer.extra_instructions == "focused"
        assert reviewer_calls == [
            (
                "https://example/pr/1",
                ["-i"],
                "fake-ai",
                (ReviewModelSelection("terra", "gpt-5.6-terra", "low"),),
            )
        ]
    finally:
        _restore_sections(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "enabled", "message"),
    [
        ("/review opus+high", False, "overrides are disabled"),
        ("/review unknown+high", True, "Unknown model alias"),
        ("/review opus++high", True, "Malformed model selector"),
        ("/review opus+extreme", True, "Unsupported reasoning effort"),
        ("/review anthropic/claude-opus-5+high", True, "Raw model identifier"),
        ("/review opus+high opus+high", True, "Duplicate model selector"),
        (
            "/review opus+none opus+minimal opus+low opus+medium opus+high",
            True,
            "Too many model selectors",
        ),
    ],
)
async def test_invalid_selector_comments_do_not_construct_a_reviewer(monkeypatch, command, enabled, message):
    snapshot = _snapshot_sections("CONFIG", "PR_REVIEWER")
    published_comments = []

    class _Provider:
        def publish_comment(self, body):
            published_comments.append(body)

    class _Reviewer:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("invalid selectors must not construct the reviewer or call a model")

    try:
        _replace_section_values("CONFIG", RESPONSE_LANGUAGE="en-us")
        _replace_section_values(
            "PR_REVIEWER",
            ENABLE_COMMAND_MODEL_OVERRIDES=enabled,
            COMMAND_MODEL_ALIASES={"opus": "anthropic/claude-opus-5"},
        )
        monkeypatch.setattr(pr_agent_module, "apply_repo_settings", lambda _pr_url: None)
        monkeypatch.setattr(pr_agent_module, "get_git_provider_with_context", lambda _pr_url: _Provider())
        monkeypatch.setitem(pr_agent_module.command2class, "review", _Reviewer)

        handled = await pr_agent_module.PRAgent()._handle_request("https://example/pr/1", command)

        assert handled is False
        assert len(published_comments) == 1
        assert message in published_comments[0]
    finally:
        _restore_sections(snapshot)
