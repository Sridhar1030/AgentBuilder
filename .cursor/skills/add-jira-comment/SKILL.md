---
name: jira-daily-update
description: >-
  Post a daily work summary as a Jira comment by collecting GitHub commits/PRs
  and Cursor chat history for the day. Use when the user asks to update a Jira
  ticket with today's work, log daily progress on a ticket, post a Jira comment
  with what was done, or runs /jira-daily-update.
---

# Jira Daily Update

Collect today's GitHub activity and Cursor chat history, then post a concise
comment on a Jira ticket summarizing what was accomplished.

## Usage

```
/jira-daily-update <JIRA-KEY> <owner/repo> [branch]
```

- `JIRA-KEY` — required (e.g. `RHOAIENG-12345`)
- `owner/repo` — required GitHub repository (e.g. `openshift/kubeflow`)
- `branch` — optional; defaults to the repo's default branch

If the user omits arguments, ask for them before proceeding.

## Step-by-Step

### 1. Resolve identities

- Call GitHub MCP `get_me` to get the authenticated username.
- Call Jira MCP `getAccessibleAtlassianResources` to get the `cloudId`.
- Call Jira MCP `getJiraIssue` with `cloudId` and `issueIdOrKey` (the JIRA-KEY)
  to confirm the ticket exists. Use `responseContentFormat: "markdown"`.
  Extract the issue summary for context.

### 2. Collect GitHub activity (today only)

Compute today's date as `YYYY-MM-DD`.

**Commits:**
- Call GitHub MCP `list_commits` with `owner`, `repo`, `sha` (branch if given),
  and `author` set to the GitHub username from step 1.
- Filter to commits where `commit.author.date` starts with today's date.
- Keep: commit message (first line) and SHA (short, first 7 chars).
- If no commits match today's date exactly, record zero commits — do NOT include
  commits from previous days.

**Pull Requests:**
- Call GitHub MCP `search_pull_requests` with query:
  `repo:<owner>/<repo> author:<username> updated:>=<YYYY-MM-DD>`
- Keep: PR title, number, state (open/closed/merged), URL.

### 3. Collect Cursor chat history (today only)

- List `~/.cursor/projects/**/agent-transcripts/` using `ls -lt` and keep only
  directories whose modification timestamp matches **today's date exactly**.
- For each matching transcript file, read JSONL lines and extract user messages:
  - Prefer text inside `<user_query>...</user_query>` tags.
  - Fall back to the first 200 chars of user message text.
- A message only counts as today's activity if the **transcript file was modified
  today** — never pull messages from transcripts modified on earlier dates.
- Distill into a short list of what the user worked on / asked the agent to do.
- Deduplicate similar entries. Aim for 3-5 distinct activity bullets.
- Ignore meta/skill invocations (e.g. `/jira-daily-update`, `/update-internship-tasks`,
  single-word messages, or messages that are only tool scaffolding).

### 4. Gate check — verify real activity exists TODAY

Before composing any comment, confirm that **at least one** of the following is true:
- There is at least one GitHub commit authored **today**.
- There is at least one PR updated **today**.
- There is at least one transcript file modified **today** containing substantive
  user messages (coding questions, feature work, debugging, design decisions).

**If none of these are true → STOP. Do NOT post a Jira comment. Tell the user:**
> "No activity found for today (<YYYY-MM-DD>). No Jira comment was posted."

**CRITICAL:** Never recycle or restate work from previous days as if it happened
today. If the only evidence is old commits or old transcripts, that is zero
activity for today — stop and report it.

### 5. Compose the Jira comment

Merge all sources into a concise progress update. Use this template:

```
**Progress update — <YYYY-MM-DD> (AI generated)**

- <bullet 1: what was done, with PR/commit link if applicable>
- <bullet 2>
- <bullet 3>
- ...
```

Rules:
- 3-5 bullets max. Each bullet is one sentence.
- Lead with the most impactful work.
- Inline GitHub links where relevant: `[PR #123](https://github.com/owner/repo/pull/123)`.
- Combine related commits + chat activity into a single bullet when they describe the same work.
- Do NOT list raw commit SHAs unless there's no PR to link.
- Do NOT include filler like "continued working on…" — be specific.
- Only include work that happened **today**. Never summarise prior days' work.
- If no activity found for a source, omit it silently.

### 6. Post to Jira

- Call Jira MCP `addCommentToJiraIssue` with:
  - `cloudId` from step 1
  - `issueIdOrKey` — the JIRA-KEY
  - `commentBody` — the composed comment
  - `contentFormat: "markdown"`
- Confirm success and show the user the posted comment text.

### 7. Show output in chat

Always display the final comment in chat so the user can review what was posted.
Format it clearly with a heading like:

```
✅ Comment posted to <JIRA-KEY>:
<the comment body>
```

## Failure Handling

- If Jira MCP is unavailable or auth fails → output the comment in chat and
  instruct the user to paste it manually.
- If GitHub MCP is unavailable → skip GitHub data, build comment from chat
  history only.
- If no transcripts found for today → skip chat data, build comment from
  GitHub only.
- If both sources are empty for today → inform the user: "No activity found for today."
- If the Jira ticket key is invalid → report the error and stop.

## Example

User runs: `/jira-daily-update RHOAIENG-53408 openshift/kubeflow release-1.9`

Posted comment:

```
**Progress update — 2026-04-01**

- Rebased SDK PR #368 on latest upstream and resolved merge conflicts
- Added unit tests for the new TrainingClient checkpoint API
- Investigated nvcc build failure blocking RHOAIENG-56094 ([PR #705](https://github.com/openshift/kubeflow/pull/705))
```
