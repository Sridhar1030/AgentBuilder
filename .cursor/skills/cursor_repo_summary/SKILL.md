---
name: daily-cursor-summary
description: >-
  Run cursor_daily_digest.py from ~/cursor-daily-digest to scan
  ~/.cursor/projects/**/agent-transcripts/ and write
  ~/.cursor/daily-summaries/YYYY-MM-DD.md. Fully local Python, no APIs.
  Use for daily Cursor digest, cursor_daily_digest, or /daily-summary workflows.
---

# Daily Cursor Digest

Standalone tool at `~/cursor-daily-digest`. Scans agent transcripts and produces a daily markdown summary.

## Run

```bash
python3 ~/cursor-daily-digest/cursor_daily_digest.py
```

Or:

```bash
~/cursor-daily-digest/run.sh
```

**Wider window** (e.g. last 7 days):

```bash
python3 ~/cursor-daily-digest/cursor_daily_digest.py --hours 168
```

**View today's summary:**

```bash
cat ~/.cursor/daily-summaries/$(date +%Y-%m-%d).md
```

## Details

- **Input**: `~/.cursor/projects/<workspace>/agent-transcripts/<id>/<id>.jsonl`
- **Filter**: transcript file mtime within last N hours (`--hours`, default 24)
- **Output**: `~/.cursor/daily-summaries/<today>.md`
