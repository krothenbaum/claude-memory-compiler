#!/usr/bin/env bash
# Print the steps to wire claude-memory-compiler into Claude Code globally.
# Auto-detects this repo's path so it works for any clone location.
# This script writes nothing — it only prints instructions you run yourself.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELL_RC=""
case "${SHELL:-}" in
  */zsh) SHELL_RC="${ZDOTDIR:-$HOME}/.zshrc" ;;
  */bash) SHELL_RC="$HOME/.bashrc" ;;
  *) SHELL_RC="(your shell's startup file, e.g. ~/.zshrc, ~/.bashrc, ~/.profile)" ;;
esac

cat <<EOF
=== claude-memory-compiler global setup ===

Detected repo at: $REPO_ROOT

Two manual steps:

1) Export CLAUDE_MEMORY_HOME so Claude Code hooks can find this repo.
   Add this line to $SHELL_RC:

     export CLAUDE_MEMORY_HOME="$REPO_ROOT"

   Then reload your shell (or open a new terminal):

     source $SHELL_RC

2) Merge this repo's .claude/settings.json into your user-global settings:

     ~/.claude/settings.json

   Specifically, copy the "hooks" object. If you already have hooks in
   ~/.claude/settings.json, merge them — don't overwrite.

After both steps, every Claude Code session (in any directory) will
capture into $REPO_ROOT/daily/ and compile into $REPO_ROOT/knowledge/.

Note: Claude Code hooks inherit the env vars of the terminal that
launched 'claude'. If you run Claude Code from a fresh terminal where
CLAUDE_MEMORY_HOME is exported, the hooks will see it.
EOF
