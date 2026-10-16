# AI Support Ticket Triage

A FastAPI service that accepts customer support requests, uses an LLM to write a short summary and classify each ticket by **category** and **priority**, stores everything in a SQL database, and serves a dashboard for reviewing tickets and model output.

**Stack:** Python, FastAPI, SQLAlchemy, SQLite / PostgreSQL, Anthropic or OpenAI API, vanilla JS dashboard, pytest, Docker.

## Features

- **LLM triage on intake.** Each new ticket is sent to the model with a strict JSON prompt: `summary`, `category`, `priority`, `confidence`.
- **Validation and fallback.** Model output is parsed and checked against a Pydantic schema (allowed labels, length limits, confidence threshold). Malformed, off-schema, or low-confidence responses are retried once, then replaced by a keyword-based rule engine so a ticket is never left untriaged.
- **Auditable results.** Every model run is stored in `triage_results` with its source (`llm` / `fallback`), model name, confidence, raw output, and any error.
- **Queue management.** Filter and search tickets, assign them to agents, change status/priority/category, and pull a priority-ordered work queue. Every change is logged in `status_updates`.
- **Dashboard.** Lists tickets with filters, shows the AI summary next to the original message, the full model-run history, and lets an agent override labels or re-run triage.

## Data model

| Table | Purpose |
|---|---|
| `tickets` | Request content plus current summary, category, priority, status, assignee |
| `triage_results` | One row per LLM/fallback run (summary, labels, confidence, raw output, error) |
| `status_updates` | Change log for status, assignee, priority, and category edits |

Categories: `billing`, `technical`, `account`, `bug`, `feature_request`, `general`
Priorities: `urgent`, `high`, `medium`, `low`
Statuses: `open`, `in_progress`, `resolved`, `closed`

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/tickets` | Create a ticket and triage it |
| `GET` | `/tickets` | List with filters: `status`, `priority`, `category`, `assignee`, `unassigned`, `q`, `sort`, `limit`, `offset` |
| `GET` | `/tickets/{id}` | Ticket detail with model runs and change history |
| `PATCH` | `/tickets/{id}` | Update status / priority / category (with optional note) |
| `POST` | `/tickets/{id}/assign` | Assign or unassign; auto-moves `open` → `in_progress` |
| `POST` | `/tickets/{id}/retriage` | Re-run the model on an existing ticket |
| `DELETE` | `/tickets/{id}` | Delete a ticket |
| `GET` | `/queue` | Active tickets ordered by priority, then age |
| `GET` | `/stats` | Counts by status/priority/category, unassigned count, fallback rate |
| `GET` | `/health` | Service status and whether an LLM is configured |

Interactive docs at `/docs` once the server is running.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add an API key
export ANTHROPIC_API_KEY=...    # or OPENAI_API_KEY=...
uvicorn app.main:app --reload
python seed.py                  # in another terminal: load sample tickets
```

Open http://localhost:8000 for the dashboard. Without an API key the app still runs, using rule-based triage only.

## Run with Docker + PostgreSQL

```bash
ANTHROPIC_API_KEY=... docker compose up --build
python seed.py
```

## Tests

```bash
pytest -q
```

Tests use a fake LLM client to cover successful triage, retry after bad output, fallback on API errors or no key, label normalization, rejection of invalid/low-confidence output, filtering, queue ordering, assignment, status history, and stats.

## Project layout

```
app/
  main.py          FastAPI routes
  triage.py        LLM clients, output validation, retry and rule-based fallback
  models.py        SQLAlchemy tables
  schemas.py       Pydantic request/response and LLM output schemas
  database.py      Engine and session setup
  static/dashboard.html
tests/test_api.py
seed.py            Sample data loader
```
