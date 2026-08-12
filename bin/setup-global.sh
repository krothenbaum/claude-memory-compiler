#!/usr/bin/env bash
# Print opt-in merge steps for Claude Code and Codex. This script writes nothing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf -v REPO_ROOT_SHELL '%q' "$REPO_ROOT"
SHELL_RC=""
case "${SHELL:-}" in
  */zsh) SHELL_RC="${ZDOTDIR:-$HOME}/.zshrc" ;;
  */bash) SHELL_RC="$HOME/.bashrc" ;;
  *) SHELL_RC="(your shell startup file, such as ~/.zshrc or ~/.bashrc)" ;;
esac

cat <<EOF
=== claude-memory-compiler global setup ===

Detected repo at: $REPO_ROOT

This setup is opt-in and does not edit either agent's configuration.

1) Export the canonical memory root in $SHELL_RC:

     export AI_MEMORY_HOME=$REPO_ROOT_SHELL

   CLAUDE_MEMORY_HOME remains a deprecated fallback for existing installs.
   Reload your shell after editing the file.

2) Merge the hooks from this repository's .claude/settings.json into:

     ~/.claude/settings.json

   Preserve every unrelated setting and hook already present; do not replace
   the destination file.

3) Codex hooks require codex-cli 0.146.1 or newer. Validate the installed CLI
   and its subscription authentication:

     codex --version
     codex features list | grep '^hooks'
     codex login status

   The last command must report exactly this supported login type:

     Logged in using ChatGPT

   Stop if it reports API-key authentication, another login type, or an error.
   This project never logs in automatically and never uses API billing.

   Then merge .codex/hooks.json.example into:

     ~/.codex/hooks.json

   Preserve every unrelated setting and hook already present; do not replace
   the destination file. The example remains inert until you merge it.

Hooks inherit environment variables from the terminal that launches Claude
Code or Codex. Start each agent from a fresh shell where AI_MEMORY_HOME is set.
EOF
