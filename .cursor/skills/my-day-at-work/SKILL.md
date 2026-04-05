---
name: my-day-at-work
description: >-
  Build a concise 24h daily work status from Cursor transcripts, Git log,
  GitHub PRs/issues, and Jira tickets, then append it to a shared Google Doc.
  Use when the user asks for a daily status, daily summary, work update,
  standup notes, my-day-at-work, or end-of-day report.
---

# My Day At Work

Create a concise team-friendly daily status for the last 24 hours, then append it to:
[daily work status ai poc](https://docs.google.com/document/d/1lAKPjyxmK1LKS1LgfPCueLHFe1sSUBy4hkohi-oZeNM/edit?tab=t.0)

## Objective

- This command is for whole-team use.
- Build one short status using Cursor, Git, and Jira history.
- Prefer complete URLs in bullets for GitHub/Jira references.
- Work with environment-specific MCP server names by discovering tools dynamically.

## Inputs

- Resolve `name` in this order:
  1. command argument
  2. Jira assignee/current user display name (if Jira is available)
  3. GitHub profile display name or login (if GitHub is available)
  4. ask user

## Data Sources

- Cursor transcript source: `~/.cursor/projects/**/agent-transcripts/**/*.jsonl`
- GitHub MCP tools (optional): discover a server that exposes GitHub issue/PR search tools
- Jira MCP tools (optional): discover a server that exposes Jira JQL search tools
- Google Docs MCP tools (optional): discover a server that exposes doc read/write tools
- MCP descriptors root: `~/.cursor/projects/<workspace>/mcps/*/tools/*.json`

## Collection Window

- Last 24 hours from now.

## Step-by-Step

1. Discover MCP servers and tools dynamically (required before MCP calls):
   - Do not assume fixed server names like `user-github`, `plugin-atlassian-atlassian`, or `user-google-service`.
   - Inspect MCP tool descriptors under:
     - `~/.cursor/projects/<workspace>/mcps/*/tools/*.json`
   - Map required capabilities to actual servers:
     - Google Docs: `get_doc`, `append_to_doc`, `replace_in_doc` (or equivalent)
     - GitHub: `search_issues` (or equivalent issue/PR search tool)
     - Jira: `searchJiraIssuesUsingJql` (or equivalent JQL search tool)
   - If a server exposes `mcp_auth`, authenticate before using its tools.

2. Collect Cursor history directly from transcripts (no file generation):
   - Read transcript files touched in last 24h from:
     - `~/.cursor/projects/**/agent-transcripts/**/*.jsonl`
   - Extract user intent with this core idea:
     - Prefer `<user_query>...</user_query>` text when present.
     - Else use user message text, normalized to one line.
   - Keep top 3-5 concise Cursor bullets.
   - If empty, use: `- No notable activity captured in last 24h.`

3. Collect Git history (local):
   - Run `git log --since="24 hours ago"` in active repo.
   - Create concise Git bullets from commits.
   - Include full commit/PR/issue links when known.

4. Collect GitHub history (optional):
   - If a discovered GitHub-capable MCP server is available and usable, query both:
     - `is:issue involves:<github_username> updated:>=<iso_time>`
     - `is:pull-request involves:<github_username> updated:>=<iso_time>`
   - Keep top 3-5 high-signal items.
   - For each item, include short context: what changed, current state, and why it matters.
   - Always include full URL in each GitHub bullet.

5. Collect Jira history (optional):
   - If a discovered Jira-capable MCP server is available and usable:
     - JQL: `assignee = currentUser() AND updated >= -1d ORDER BY updated DESC`
   - Keep top 3-5 issues.
   - For each issue, include short context: key objective, latest status, and next step when possible.
   - Always include full Jira issue URL in each Jira bullet.

6. Build status text in this exact shape:

   date: YYYY-MM-DD
   name: \<name\>
   status summary:
   - \<cursor activity summary with brief context\>
   - \<git activity summary with brief context and link when available\>
   - \<github/jira activity summary with brief context and full link\>

   Formatting rules:
   - Keep each bullet one line.
   - Do not prefix bullets with `Cursor:`, `Git:`, or `Jira:`.
   - Keep wording precise, but include enough context to be useful to teammates.
   - Prefer slightly richer bullets over overly short fragments.
   - Prefer full links over short references.

7. Append to Google Doc when possible:
   - Target document id: `1lAKPjyxmK1LKS1LgfPCueLHFe1sSUBy4hkohi-oZeNM`
   - Placement rules (date-aware):
     - Read current doc text first (`get_doc`).
     - If `date: YYYY-MM-DD` already exists:
       - Append the new person status block under that same date section (below existing statuses for that date, not at document bottom).
     - If `date: YYYY-MM-DD` does not exist:
       - Insert the new date block at the top of the status area (newest date first), above older dates.
   - Use `replace_in_doc` for controlled placement when needed.
   - Use `append_to_doc` only as a fallback when structured placement is not possible.

8. Verify write when append succeeds:
   - Use `get_doc` and confirm the appended block exists.

## Failure Handling

- If any MCP server is missing, unavailable, or unauthenticated, ignore it and continue.
- Do not show hard errors for missing GitHub/Jira/Google MCP.
- If Google Doc write is not allowed or append fails, provide the final status block in chat.
- If date-aware insertion cannot be completed safely, do not corrupt formatting; return the status block in chat.
- If a source has no data, keep concise fallback bullets:
  - `- No notable Cursor activity captured in last 24h.`
  - `- No notable Git activity captured in last 24h.`
  - `- No notable Jira/GitHub updates captured in last 24h.`
