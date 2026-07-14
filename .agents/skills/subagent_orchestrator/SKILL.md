---
name: subagent-orchestrator
description: Provides guidelines, best practices, and patterns for defining, invoking, managing, and communicating with Antigravity subagents. Use this skill when you need to delegate tasks to subagents, manage parallel background agents, or design a multi-agent workflow.
---

# Subagent Orchestrator Skill

This skill provides guidelines and patterns for using Google Antigravity's subagent orchestration features.

---

## 1. Subagent Lifecycle Management

### Defining Subagents
* Use the `define_subagent` tool to register a new type of subagent.
* Assign a specific `name`, `description`, and a tailored `system_prompt`.
* Select appropriate capability groups:
  - `enable_write_tools`: Required if the subagent needs to edit code or run terminal commands.
  - `enable_subagent_tools`: Allow the subagent to spawn its own child subagents.
  - `enable_mcp_tools`: Allow the subagent to access Model Context Protocol tools.

### Invoking Subagents
* Use the `invoke_subagent` tool, specifying the `TypeName`, `Role` (job title), and a clear, detailed `Prompt` of what the subagent must execute.
* Spawning can be configured to use `inherit` (default), `branch` (isolated duplicate repository), or `share` (shared git worktree).

### Managing and Cleaning Up
* Run `manage_subagents` with the `list` action to get the status and conversation IDs of all running subagents.
* Run `manage_subagents` with the `kill` action (specifying `ConversationIds`) or `kill_all` to terminate completed or idle subagents. **Always terminate subagents when their job is done to free up resources.**

---

## 2. Multi-Agent Communication

### Exchanging Messages
* Use the `send_message` tool to communicate with active subagents. Always pass the subagent's `conversationID` as the recipient.
* **CRITICAL**: Do NOT use the `send_message` tool to communicate with the user. Use normal chat response text for user interaction.

### Reactive Wakeup (No Polling)
* The Antigravity runtime automatically resumes parent agent execution when a subagent sends a message.
* **Never call `manage_subagents` or read transcript logs in a loop to wait for completion.**
* Simply stop calling tools (end your turn) after launching or messaging a subagent, and wait for the system to wake you up.

---

## 3. Best Practices & Design Patterns

* **Single Responsibility Principle:** Create specific subagents for specific, isolated tasks (e.g., a `codebase_researcher` to find patterns, or a `prompt_optimizer` to edit docs) rather than one agent for everything.
* **Context Preservation:** Pass all necessary context (file paths, rules, recent changes) in the subagent's initial prompt so it does not waste turns querying the parent.
* **Transcript Reading for Debugging:** If a subagent reports a failure, read its untruncated transcript log at `<appDataDir>/brain/<conversation-id>/.system_generated/logs/transcript_full.jsonl` to trace the exact error sequence.
