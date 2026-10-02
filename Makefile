# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Activate the repo's AI coding-assistant assets from skills_studio/.
#
#   make claude     link skills_studio/ into .claude/   (Claude Code)
#   make codex      link skills_studio/ into .agents/ + .codex/   (Codex)
#   make clean-ai   remove everything the two targets above created
#
# skills_studio/ is the version-controlled source of truth; .claude/, .codex/ and
# .agents/ are gitignored per-developer activation layers. Assets are symlinked,
# never copied, so a `git pull` updates them and your edits land in tracked files.

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
STUDIO    := $(REPO_ROOT)/skills_studio
AGENT_TO_TOML := $(STUDIO)/tools/md_agent_to_codex_toml.py
PYTHON    ?= python3

.PHONY: help claude codex clean-ai

help:
	@echo "make claude     activate skills_studio assets for Claude Code (.claude/)"
	@echo "make codex      activate skills_studio assets for Codex (.agents/, .codex/)"
	@echo "make clean-ai   remove the generated .claude/, .codex/ and .agents/ assets"

# Claude Code reads Markdown + YAML frontmatter for all three asset types, so every
# directory links across as-is. `ln -sfn` is idempotent and re-points a stale link.
# An existing real directory is left alone rather than clobbered — link per-asset in
# that case (see AGENTS.md).
claude:
	@mkdir -p $(REPO_ROOT)/.claude
	@for kind in skills agents commands; do \
	    src=$(STUDIO)/$$kind; dst=$(REPO_ROOT)/.claude/$$kind; \
	    [ -d "$$src" ] || { echo "skip  .claude/$$kind (no skills_studio/$$kind)"; continue; }; \
	    if [ -e "$$dst" ] && [ ! -L "$$dst" ]; then \
	        echo "SKIP  .claude/$$kind exists and is not a symlink — link assets individually"; \
	        continue; \
	    fi; \
	    ln -sfn "$$src" "$$dst" && echo "link  .claude/$$kind -> skills_studio/$$kind"; \
	done
	@echo "Done. Start Claude Code from $(REPO_ROOT) to pick the assets up."

# Codex differs from Claude in two ways that rule out a blanket symlink:
#   - skills live under .agents/skills (not .codex/skills) but use the same
#     SKILL.md + name/description format, so the directory links across directly.
#   - subagents are standalone TOML (.codex/agents/*.toml) with a
#     `developer_instructions` key, so they are generated from the Markdown
#     definitions instead of linked. Regenerate after editing the source .md.
# Codex custom prompts have no documented on-disk project-level location, so
# skills_studio/commands is not wired up here; invoke the equivalent skill instead.
codex:
	@if [ -d $(STUDIO)/skills ]; then \
	    mkdir -p $(REPO_ROOT)/.agents; \
	    dst=$(REPO_ROOT)/.agents/skills; \
	    if [ -e "$$dst" ] && [ ! -L "$$dst" ]; then \
	        echo "SKIP  .agents/skills exists and is not a symlink — link assets individually"; \
	    else \
	        ln -sfn $(STUDIO)/skills "$$dst" && echo "link  .agents/skills -> skills_studio/skills"; \
	    fi; \
	fi
	@if [ -d $(STUDIO)/agents ]; then \
	    mkdir -p $(REPO_ROOT)/.codex/agents; \
	    for md in $(STUDIO)/agents/*.md; do \
	        [ -e "$$md" ] || continue; \
	        out=$(REPO_ROOT)/.codex/agents/$$(basename "$$md" .md).toml; \
	        $(PYTHON) $(AGENT_TO_TOML) "$$md" "$$out" || exit 1; \
	        echo "gen   .codex/agents/$$(basename "$$out") (from $$(basename "$$md"))"; \
	    done; \
	fi
	@if [ -d $(STUDIO)/commands ]; then \
	    echo "note  skills_studio/commands is Claude-only — no project-level Codex equivalent"; \
	fi
	@echo "Done. Start Codex from $(REPO_ROOT) to pick the assets up."

# Only removes what `make claude` / `make codex` create: the asset symlinks and the
# generated agent TOMLs. Any other local content under .claude/ or .codex/ (your
# settings, session state) is left untouched, and the directories are removed only
# if emptied.
clean-ai:
	@for kind in skills agents commands; do \
	    dst=$(REPO_ROOT)/.claude/$$kind; \
	    if [ -L "$$dst" ]; then rm -f "$$dst" && echo "rm    .claude/$$kind"; fi; \
	done
	@if [ -L $(REPO_ROOT)/.agents/skills ]; then \
	    rm -f $(REPO_ROOT)/.agents/skills && echo "rm    .agents/skills"; \
	fi
	@if [ -d $(REPO_ROOT)/.codex/agents ] && [ -d $(STUDIO)/agents ]; then \
	    for md in $(STUDIO)/agents/*.md; do \
	        [ -e "$$md" ] || continue; \
	        out=$(REPO_ROOT)/.codex/agents/$$(basename "$$md" .md).toml; \
	        if [ -f "$$out" ]; then rm -f "$$out" && echo "rm    .codex/agents/$$(basename "$$out")"; fi; \
	    done; \
	fi
	@rmdir $(REPO_ROOT)/.claude $(REPO_ROOT)/.agents $(REPO_ROOT)/.codex/agents \
	    $(REPO_ROOT)/.codex 2>/dev/null || true
	@echo "Done."
