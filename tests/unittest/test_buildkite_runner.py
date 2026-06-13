import copy

import pytest

import pr_agent.servers.buildkite_runner as buildkite_runner
from pr_agent.config_loader import get_settings


@pytest.mark.parametrize("repo_url,expected", [
    ("git@github.com:org/repo.git", ("github.com", "org/repo")),
    ("https://github.com/org/repo.git", ("github.com", "org/repo")),
    ("https://github.com/org/repo", ("github.com", "org/repo")),
    ("ssh://git@bitbucket.org/org/repo.git", ("bitbucket.org", "org/repo")),
    ("git@gitlab.com:group/subgroup/repo.git", ("gitlab.com", "group/subgroup/repo")),
    ("git://github.com/org/repo.git", ("github.com", "org/repo")),
])
def test_parse_repo_url_supported_forms(repo_url, expected):
    assert buildkite_runner.parse_repo_url(repo_url) == expected


@pytest.mark.parametrize("repo_url", ["", None, "github.com", "git@github.com:repo"])
def test_parse_repo_url_rejects_invalid_urls(repo_url):
    assert buildkite_runner.parse_repo_url(repo_url) is None


@pytest.mark.parametrize("repo_url,pr_number,expected", [
    ("git@github.com:org/repo.git", "42",
     ("github", "https://github.com/org/repo/pull/42")),
    ("https://bitbucket.org/org/repo.git", "7",
     ("bitbucket", "https://bitbucket.org/org/repo/pull-requests/7")),
    ("git@gitlab.com:org/repo.git", "3",
     ("gitlab", "https://gitlab.com/org/repo/-/merge_requests/3")),
])
def test_build_pr_url_per_host(repo_url, pr_number, expected):
    assert buildkite_runner.build_pr_url(repo_url, pr_number) == expected


def test_build_pr_url_returns_none_for_unknown_host():
    assert buildkite_runner.build_pr_url("git@git.example.com:org/repo.git", "1") is None


@pytest.mark.asyncio
async def test_run_action_skips_non_pull_request_builds(monkeypatch, capsys):
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")

    await buildkite_runner.run_action()

    assert "Not a pull request build" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_action_returns_when_repo_is_missing(monkeypatch, capsys):
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "42")
    monkeypatch.delenv("BUILDKITE_REPO", raising=False)
    monkeypatch.delenv("PR_AGENT_PR_URL", raising=False)

    await buildkite_runner.run_action()

    assert "BUILDKITE_REPO not set" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_action_returns_for_unknown_git_host(monkeypatch, capsys):
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "42")
    monkeypatch.setenv("BUILDKITE_REPO", "git@git.example.com:org/repo.git")
    monkeypatch.delenv("PR_AGENT_PR_URL", raising=False)

    await buildkite_runner.run_action()

    assert "Set PR_AGENT_PR_URL" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_action_invokes_enabled_auto_tools(monkeypatch):
    settings = get_settings()
    original_is_auto_command = settings.config.get("is_auto_command", False)
    original_final_update_message = settings.pr_description.final_update_message
    original_git_provider = settings.config.git_provider
    had_github_settings = "GITHUB" in settings
    original_github_settings = copy.deepcopy(settings.get("GITHUB", None))

    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "42")
    monkeypatch.setenv("BUILDKITE_REPO", "git@github.com:org/repo.git")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.delenv("PR_AGENT_PR_URL", raising=False)
    monkeypatch.delenv("CONFIG__GIT_PROVIDER", raising=False)
    monkeypatch.setattr(buildkite_runner, "apply_repo_settings", lambda pr_url: None)

    def fake_get_setting_or_env(key, default=None):
        values = {
            "BUILDKITE_CONFIG.AUTO_DESCRIBE": True,
            "BUILDKITE_CONFIG.AUTO_REVIEW": False,
            "BUILDKITE_CONFIG.AUTO_IMPROVE": True,
        }
        return values.get(key, default)

    monkeypatch.setattr(buildkite_runner, "get_setting_or_env", fake_get_setting_or_env)
    runs = []

    class FakeTool:
        name = "base"

        def __init__(self, pr_url):
            self.pr_url = pr_url

        async def run(self):
            runs.append((self.name, self.pr_url))

    class FakeDescription(FakeTool):
        name = "describe"

    class FakeReviewer(FakeTool):
        name = "review"

    class FakeSuggestions(FakeTool):
        name = "improve"

    monkeypatch.setattr(buildkite_runner, "PRDescription", FakeDescription)
    monkeypatch.setattr(buildkite_runner, "PRReviewer", FakeReviewer)
    monkeypatch.setattr(buildkite_runner, "PRCodeSuggestions", FakeSuggestions)

    try:
        await buildkite_runner.run_action()

        assert runs == [
            ("describe", "https://github.com/org/repo/pull/42"),
            ("improve", "https://github.com/org/repo/pull/42"),
        ]
        assert settings.get("GITHUB.USER_TOKEN") == "token"
        assert settings.config.git_provider == "github"
    finally:
        settings.config.is_auto_command = original_is_auto_command
        settings.pr_description.final_update_message = original_final_update_message
        settings.config.git_provider = original_git_provider
        if had_github_settings:
            settings.set("GITHUB", original_github_settings)
        else:
            settings.unset("GITHUB", force=True)
