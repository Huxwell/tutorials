# Preferences

## Commands that require sudo
- Claude Code has no terminal, so `sudo` can't prompt for a password here (and `!` commands fail the same way). Don't run sudo commands yourself.
- Give the user the command to run in their own terminal, with output piped so it's shown and saved for you to read:
  `sudo <command> 2>&1 | tee ~/out.txt`
- Then read `~/out.txt` instead of asking the user to copy the output.
- Add `-y` to apt commands, since piping can break the interactive `[Y/n]` prompt.

## Git
- Never add `Co-Authored-By` (or any Claude attribution) lines to commit messages or PR descriptions.

## Configs
- Never change my configs (`~/.ssh/*`, VS Code settings, git config, dotfiles, etc.) without explicit confirmation first. Propose the change and wait.

## Answers
- Prefer short answers. Beware that the user may only read part of your answer if it's long and a debugging loop is fast&intense.

## Python environments
- Use `uv` (installed at `~/.local/bin/uv`) for venvs and packages.
- Put venvs in visible, named folders (e.g. `pytorch/torch_venv`), not hidden ones like `.venv`.
- GPU is a GTX 1080 (Pascal, sm_61): use PyTorch `cu126` wheels (`--index-url https://download.pytorch.org/whl/cu126`); cu128+ dropped Pascal.
