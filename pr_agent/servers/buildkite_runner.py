import asyncio
import os
from typing import Optional, Tuple, Union

from pr_agent.config_loader import get_settings
from pr_agent.git_providers.utils import apply_repo_settings
from pr_agent.log import get_logger
from pr_agent.tools.pr_code_suggestions import PRCodeSuggestions
from pr_agent.tools.pr_description import PRDescription
from pr_agent.tools.pr_reviewer import PRReviewer

# Maps a git host to the pr-agent git provider and the pull request URL format
PR_URL_FORMATS = {
    "github.com": ("github", "https://github.com/{slug}/pull/{number}"),
    "bitbucket.org": ("bitbucket", "https://bitbucket.org/{slug}/pull-requests/{number}"),
    "gitlab.com": ("gitlab", "https://gitlab.com/{slug}/-/merge_requests/{number}"),
}


def is_true(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == 'true'
    return False


def get_setting_or_env(key: str, default: Union[str, bool] = None) -> Union[str, bool]:
    try:
        value = get_settings().get(key, default)
    except AttributeError:
        value = os.getenv(key, None) or os.getenv(key.upper(), None) or os.getenv(key.lower(), None) or default
    return value


def parse_repo_url(repo_url: str) -> Optional[Tuple[str, str]]:
    """Extracts (host, "org/repo" slug) from a git remote URL.

    Supports scp-style (git@host:org/repo.git), ssh://, git://, http(s):// and
    plain host/org/repo forms.
    """
    if not repo_url:
        return None
    stripped = repo_url.strip()
    stripped = stripped.removesuffix(".git")
    for prefix in ("ssh://", "git://", "https://", "http://"):
        stripped = stripped.removeprefix(prefix)
    stripped = stripped.removeprefix("git@")

    colon_pos = stripped.find(":")
    slash_pos = stripped.find("/")
    if colon_pos != -1 and (slash_pos == -1 or colon_pos < slash_pos):
        # scp-style: host:org/repo
        host, slug = stripped[:colon_pos], stripped[colon_pos + 1:]
    else:
        host, _, slug = stripped.partition("/")

    if not host or not slug or "/" not in slug:
        return None
    return host, slug


def build_pr_url(repo_url: str, pr_number: str) -> Optional[Tuple[str, str]]:
    """Returns (git_provider, pr_url) for the build's repository, or None if the
    host is not recognized."""
    parsed = parse_repo_url(repo_url)
    if not parsed:
        return None
    host, slug = parsed
    if host not in PR_URL_FORMATS:
        return None
    provider, url_format = PR_URL_FORMATS[host]
    return provider, url_format.format(slug=slug, number=pr_number)


async def run_action():
    # Buildkite sets BUILDKITE_PULL_REQUEST to the PR number, or "false" on non-PR builds
    pr_number = os.environ.get('BUILDKITE_PULL_REQUEST', 'false')
    if not pr_number or pr_number == 'false':
        print("Not a pull request build (BUILDKITE_PULL_REQUEST is not set), skipping")
        return

    provider = None
    pr_url = os.environ.get('PR_AGENT_PR_URL')
    if not pr_url:
        repo_url = os.environ.get('BUILDKITE_REPO')
        if not repo_url:
            print("BUILDKITE_REPO not set")
            return
        derived = build_pr_url(repo_url, pr_number)
        if not derived:
            print(f"Cannot derive a pull request URL from BUILDKITE_REPO='{repo_url}'. "
                  f"Set PR_AGENT_PR_URL for repositories hosted outside github.com/bitbucket.org/gitlab.com")
            return
        provider, pr_url = derived

    # Set the git provider derived from the repository host, unless explicitly configured
    if provider and not os.environ.get('CONFIG__GIT_PROVIDER') and not os.environ.get('CONFIG.GIT_PROVIDER'):
        get_settings().set("CONFIG.GIT_PROVIDER", provider)

    # Map common CI secrets to their pr-agent settings
    OPENAI_KEY = os.environ.get('OPENAI_KEY') or os.environ.get('OPENAI_API_KEY')
    if OPENAI_KEY:
        get_settings().set("OPENAI.KEY", OPENAI_KEY)
    ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY')
    if ANTHROPIC_KEY:
        get_settings().set("ANTHROPIC.KEY", ANTHROPIC_KEY)
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
    if GITHUB_TOKEN:
        get_settings().set("GITHUB.USER_TOKEN", GITHUB_TOKEN)
        get_settings().set("GITHUB.DEPLOYMENT_TYPE", "user")

    if get_settings().get("CONFIG.GIT_PROVIDER") == "github" and not get_settings().get("GITHUB.USER_TOKEN", None):
        print("GITHUB_TOKEN not set")
        return

    try:
        get_logger().info("Applying repo settings")
        apply_repo_settings(pr_url)
    except Exception as e:
        get_logger().info(f"buildkite runner: failed to apply repo settings: {e}")

    auto_describe = get_setting_or_env("BUILDKITE_CONFIG.AUTO_DESCRIBE", None)
    auto_review = get_setting_or_env("BUILDKITE_CONFIG.AUTO_REVIEW", None)
    auto_improve = get_setting_or_env("BUILDKITE_CONFIG.AUTO_IMPROVE", None)

    get_settings().config.is_auto_command = True
    get_settings().pr_description.final_update_message = False  # No final update message when auto_describe is enabled
    get_logger().info(f"Running auto actions: auto_describe={auto_describe}, auto_review={auto_review}, auto_improve={auto_improve}")

    # invoke by default all three tools
    if auto_describe is None or is_true(auto_describe):
        await PRDescription(pr_url).run()
    if auto_review is None or is_true(auto_review):
        await PRReviewer(pr_url).run()
    if auto_improve is None or is_true(auto_improve):
        await PRCodeSuggestions(pr_url).run()


if __name__ == '__main__':
    asyncio.run(run_action())
