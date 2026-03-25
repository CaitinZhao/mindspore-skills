# MindSpore Skills

MindSpore development skills for AI coding agents. Route operator work, migrate models, and diagnose readiness, failure, accuracy, and performance issues with guided workflows.

Compatible with **Claude Code**, **OpenCode**, **Gemini CLI**, and **Codex**.

## Installation

### Claude Code

Register the marketplace and install:

```
/plugin marketplace add vigo999/mindspore-skills
/plugin install mscode@mindspore-skills
```

Then use slash command:

```
/mscode:migrate
/mscode:model-agent
/mscode:api-helper
/mscode:operator-agent
/mscode:readiness-agent
/mscode:accuracy-agent
/mscode:failure-agent
/mscode:algorithm-agent
/mscode:performance-agent
```

### OpenCode

Clone to your global config:

```bash
git clone https://github.com/vigo999/mindspore-skills.git ~/.config/opencode/mindspore-skills
ln -s ~/.config/opencode/mindspore-skills/skills ~/.config/opencode/skills
ln -s ~/.config/opencode/mindspore-skills/commands ~/.config/opencode/commands
```

Or for a specific project:

```bash
git clone https://github.com/vigo999/mindspore-skills.git .opencode
```

Then in OpenCode:

```
/operator-agent
```

See [OpenCode Skills docs](https://opencode.ai/docs/skills) for more details.

### Gemini CLI

Install as an extension:

```bash
gemini extensions install https://github.com/vigo999/mindspore-skills.git --consent
```

Or from local clone:

```bash
git clone https://github.com/vigo999/mindspore-skills.git
gemini extensions install ./mindspore-skills --consent
```

See [Gemini CLI extensions docs](https://geminicli.com/docs/extensions/) for more details.

### Codex

Clone to your project root:

```bash
git clone https://github.com/vigo999/mindspore-skills.git .mindspore-skills
```

Codex reads `AGENTS.md` automatically. Verify with:

```bash
codex "Summarize the MindSpore skills available."
```

See [Codex AGENTS guide](https://developers.openai.com/codex/guides/agents-md) for more details.

## Available Skills

### Operator Development

| Skill | Description |
|-------|-------------|
| `api-helper` | Find API call chains and operator wiring in MindSpore codebase |
| `operator-agent` | Route and build `torch` or `mindspore` operators through custom-access or native-framework integration, with MindSpore API-resolution and `op_info` verification support |

### Model Migration

| Skill | Description |
|-------|-------------|
| `model-agent` | Top-level model migration entry that analyzes the source repo, selects the correct migration route, and verifies the result |

### Diagnosis and Optimization

| Skill | Description |
|-------|-------------|
| `accuracy-agent` | Diagnose accuracy regressions, drift, wrong results, and cross-platform mismatch after successful execution |
| `algorithm-agent` | Adapt a paper feature or released reference implementation into an existing model codebase and prepare the patch for readiness validation |
| `readiness-agent` | Analyze a local single-machine training workspace and validate pre-run readiness before training |
| `failure-agent` | Diagnose MindSpore and PTA (torch_npu) training and runtime failures with evidence-backed root-cause validation |
| `performance-agent` | Diagnose throughput, latency, memory, utilization, dataloader, and communication bottlenecks after the workload already runs |

## Available Commands

### Operator Development

| Command | Description |
|---------|-------------|
| `/api-helper` | API chain discovery workflow |
| `/operator-agent` | Operator routing and implementation workflow with custom-access or native-framework integration |

### Model Migration

| Command | Description |
|---------|-------------|
| `/migrate` | Migration router (HF/third-party), routing only |
| `/model-agent` | Top-level model migration workflow with route selection and verification |

### Diagnosis and Optimization

| Command | Description |
|---------|-------------|
| `/accuracy-agent` | Accuracy diagnosis workflow after successful execution |
| `/algorithm-agent` | Algorithm feature adaptation workflow with patch generation and readiness handoff |
| `/readiness-agent` | Single-machine training workspace readiness workflow |
| `/failure-agent` | Dual-stack failure diagnosis workflow with evidence, root-cause validation, and report output |
| `/performance-agent` | Performance diagnosis workflow with bottleneck validation and report output |

## Usage Examples

### Build an operator

```
/operator-agent

> Help me implement the linspace operator and choose the right integration path
```

### Examples

See `examples/README.md` for the current example inventory and status.

## Contract and Tests

- Architecture overview:
  - `docs/concepts/agent-architecture-overview.md`
- Contract docs:
  - `docs/concepts/skills-contract.md`
  - `docs/concepts/artifacts-and-reporting.md`
- Cross-skill contract tests: `tests/contract/`
- Skill-specific tests: `skills/<skill>/tests/`

## Repository Structure

```
mindspore-skills/
├── commands/                # Slash commands
│   ├── api-helper.md        # API chain discovery
│   ├── failure-agent.md     # Failure diagnosis
│   ├── migrate.md           # Migration router
│   └── ...
├── skills/                  # Skill definitions
│   ├── api-helper/          # MindSpore API call-chain discovery
│   ├── model-agent/         # Top-level model migration entry
│   ├── operator-agent/      # Framework operator implementation
│   ├── readiness-agent/     # Training workspace readiness and preflight
│   ├── accuracy-agent/      # Accuracy diagnosis after successful execution
│   ├── algorithm-agent/     # Feature adaptation and patch planning for existing models
│   ├── failure-agent/       # Training and runtime failure diagnosis
│   └── performance-agent/   # Performance diagnosis after the workload already runs
├── AGENTS.md                # Codex instructions
├── CLAUDE.md                # Claude-facing repository notes
└── gemini-extension.json    # Gemini CLI config
```

## Contributing

Contributions are welcome. Please submit a pull request.

When adding a new skill:
1. Add `skills/<skill-name>/SKILL.md` with matching frontmatter and directory name
2. Add a slash command in `commands/<command-name>.md` if needed
3. Update `AGENTS.md` (skill table + activation triggers)
4. Update `README.md` (skill list and commands)
5. Update `gemini-extension.json` with name/path/description
6. Update `CLAUDE.md` if Claude-facing setup or maintenance notes changed

When modifying an existing skill:
1. Update `skills/<skill-name>/SKILL.md` and any referenced files
2. Refresh `AGENTS.md` triggers if scope/keywords changed
3. Update `README.md` if descriptions or commands changed
4. Update `gemini-extension.json` if name/path/description changed

tools:
- Run `python tools/check_consistency.py` before submit
- (Optional) Install git hooks with `python tools/install_git_hooks.py`
- Tip: set up hooks with `make hooks` (see Makefile).

## License

Apache 2.0
