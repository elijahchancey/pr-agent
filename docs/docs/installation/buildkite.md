# Run as a Buildkite Step

You can run PR-Agent from a [Buildkite](https://buildkite.com) pipeline on every pull request build, using the dedicated `pragent/pr-agent:buildkite` Docker image. The image reads Buildkite's build environment (`BUILDKITE_PULL_REQUEST`, `BUILDKITE_REPO`), derives the pull request URL, and runs the `describe`, `review` and `improve` tools. Non-PR builds are skipped automatically.

1. Add a step to your `pipeline.yml`:

```yaml
steps:
  - label: ":robot_face: PR Agent"
    if: build.pull_request.id != null
    command: |
      docker run --rm \
        -e BUILDKITE_PULL_REQUEST -e BUILDKITE_REPO \
        -e GITHUB_TOKEN -e OPENAI_KEY \
        pragent/pr-agent:buildkite
```

2. Provide the secrets as environment variables on the agent (via an [agent `environment` hook](https://buildkite.com/docs/agent/v3/hooks), the [secrets plugin](https://github.com/buildkite-plugins/secrets-buildkite-plugin), or your cluster's secret storage — avoid `env` in `pipeline.yml`, which is visible in the UI):

   - `GITHUB_TOKEN`: a GitHub PAT (or GitHub App installation token) with read/write access to pull requests on the repository. Unlike GitHub Actions, Buildkite does not inject a repository token automatically.
   - `OPENAI_KEY` (or any other model key, e.g. `ANTHROPIC_API_KEY`): the API key for the model you use. See [Changing a model](../usage-guide/changing_a_model.md).

3. Make sure **Build Pull Requests** is enabled in the pipeline's Git provider settings, otherwise `BUILDKITE_PULL_REQUEST` is not populated and the step skips.

## Configuration

Every PR-Agent [configuration option](../usage-guide/configuration_options.md) can be passed as an environment variable in `SECTION__KEY` form (remember to forward it with `-e`), for example:

```yaml
    command: |
      docker run --rm \
        -e BUILDKITE_PULL_REQUEST -e BUILDKITE_REPO \
        -e GITHUB_TOKEN -e ANTHROPIC_API_KEY \
        -e CONFIG__MODEL="anthropic/claude-sonnet-4-6" \
        -e BUILDKITE_CONFIG__AUTO_DESCRIBE=false \
        pragent/pr-agent:buildkite
```

The Buildkite-specific options are:

| Variable | Default | Meaning |
|---|---|---|
| `BUILDKITE_CONFIG__AUTO_DESCRIBE` | `true` | Run the `describe` tool |
| `BUILDKITE_CONFIG__AUTO_REVIEW` | `true` | Run the `review` tool |
| `BUILDKITE_CONFIG__AUTO_IMPROVE` | `true` | Run the `improve` tool |
| `PR_AGENT_PR_URL` | derived | Explicit PR URL, required for self-hosted git servers (e.g. GitHub Enterprise; also set `GITHUB__BASE_URL`) |

Repositories hosted on github.com, bitbucket.org and gitlab.com are recognized automatically (including setting `config.git_provider`).

## Buildkite Plugin

Alternatively, use the [PR Agent Buildkite plugin](https://github.com/elijahchancey/pr-agent-buildkite-plugin), which wraps the same flow with plugin-style configuration:

```yaml
steps:
  - label: ":robot_face: PR Agent"
    if: build.pull_request.id != null
    plugins:
      - elijahchancey/pr-agent#v1.0.0:
          version: "0.36.0"
          model: "anthropic/claude-sonnet-4-6"
```

## Limitations

Comment-triggered commands (e.g. writing `/ask` in a PR comment) require webhook events and cannot be handled from a CI build. To get interactive commands alongside the automatic CI review, also deploy the [GitHub App or polling server](./github.md).
