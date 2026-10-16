import json
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_triage.db"
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app, get_llm_client
from app.triage import parse_llm_output, rule_based_triage


class FakeLLM:
    """Returns queued responses in order; raises if a response is an Exception."""

    model = "fake-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


GOOD = json.dumps(
    {"summary": "Customer was charged twice for March.", "category": "billing",
     "priority": "high", "confidence": 0.92}
)
TICKET = {"customer_email": "a@example.com", "subject": "Double charge",
          "body": "I was charged twice this month. Please refund."}


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def use_llm(fake):
    app.dependency_overrides[get_llm_client] = lambda: fake
    return fake


# ---------- validation unit tests ----------

def test_parse_accepts_json_wrapped_in_prose():
    r = parse_llm_output(f"Sure! Here you go:\n```json\n{GOOD}\n```")
    assert r.category == "billing" and r.priority == "high"


def test_parse_normalizes_labels():
    r = parse_llm_output(json.dumps({"summary": "Wants dark mode.", "category": "Feature Request",
                                     "priority": "LOW", "confidence": 0.8}))
    assert r.category == "feature_request" and r.priority == "low"


@pytest.mark.parametrize("bad", [
    "no json here",
    '{"summary": "x", "category": "billing"',                                   # truncated
    json.dumps({"summary": "Valid summary", "category": "sales", "priority": "high", "confidence": 0.9}),
    json.dumps({"summary": "Valid summary", "category": "bug", "priority": "critical", "confidence": 0.9}),
    json.dumps({"summary": "Valid summary", "category": "bug", "priority": "high", "confidence": 0.2}),
])
def test_parse_rejects_unreliable_output(bad):
    with pytest.raises(ValueError):
        parse_llm_output(bad)


def test_rule_based_triage():
    _, cat, pri = rule_based_triage("Site down", "Total outage, all users get a 500 error")
    assert cat == "bug" and pri == "urgent"
    _, cat, pri = rule_based_triage("Idea", "Feature suggestion: add support for CSV export")
    assert cat == "feature_request" and pri == "low"


# ---------- API tests ----------

def test_create_ticket_uses_llm(client):
    use_llm(FakeLLM(GOOD))
    r = client.post("/tickets", json=TICKET)
    assert r.status_code == 201
    t = r.json()
    assert (t["category"], t["priority"]) == ("billing", "high")
    assert t["triage_results"][0]["source"] == "llm"
    assert t["triage_results"][0]["confidence"] == 0.92


def test_retries_then_succeeds(client):
    fake = use_llm(FakeLLM("garbage", GOOD))
    t = client.post("/tickets", json=TICKET).json()
    assert fake.calls == 2
    assert t["triage_results"][0]["source"] == "llm"


def test_falls_back_when_llm_unreliable(client):
    import httpx
    use_llm(FakeLLM("not json", httpx.ConnectError("boom")))
    t = client.post("/tickets", json=TICKET).json()
    run = t["triage_results"][0]
    assert run["source"] == "fallback"
    assert "attempt 1" in run["error"] and "attempt 2" in run["error"]
    assert t["category"] == "billing" and t["priority"] == "high"


def test_falls_back_without_llm(client):
    use_llm(None)
    t = client.post("/tickets", json=TICKET).json()
    assert t["triage_results"][0]["source"] == "fallback"


def test_input_validation(client):
    use_llm(None)
    assert client.post("/tickets", json={**TICKET, "customer_email": "nope"}).status_code == 422
    assert client.post("/tickets", json={**TICKET, "body": ""}).status_code == 422


def test_filtering_and_queue_order(client):
    use_llm(None)
    bodies = [
        ("Question", "How do I export my data? Just wondering."),          # low
        ("Outage", "Total outage for all users right now"),                 # urgent
        ("Locked out", "I'm locked out of my account and can't log in"),     # high
    ]
    for s, b in bodies:
        client.post("/tickets", json={**TICKET, "subject": s, "body": b})

    queue = client.get("/queue").json()
    assert [t["priority"] for t in queue] == ["urgent", "high", "low"]

    assert len(client.get("/tickets", params={"priority": "urgent"}).json()) == 1
    assert len(client.get("/tickets", params={"category": "account"}).json()) == 1
    assert len(client.get("/tickets", params={"q": "export"}).json()) == 1


def test_assignment_and_status_history(client):
    use_llm(None)
    tid = client.post("/tickets", json=TICKET).json()["id"]

    t = client.post(f"/tickets/{tid}/assign", json={"assignee": "martin"}).json()
    assert t["assignee"] == "martin" and t["status"] == "in_progress"

    t = client.patch(f"/tickets/{tid}", json={"status": "resolved", "note": "refunded"}).json()
    assert t["status"] == "resolved"
    fields = [(u["field"], u["new_value"]) for u in t["status_updates"]]
    assert fields == [("assignee", "martin"), ("status", "in_progress"), ("status", "resolved")]

    assert len(client.get("/tickets", params={"assignee": "martin"}).json()) == 1
    assert client.get("/queue", params={"assignee": "martin"}).json() == []


def test_active_and_resolved_views(client):
    use_llm(None)
    ids = [client.post("/tickets", json=TICKET).json()["id"] for _ in range(3)]
    client.patch(f"/tickets/{ids[0]}", json={"status": "resolved"})
    client.patch(f"/tickets/{ids[1]}", json={"status": "closed"})

    active = client.get("/tickets", params={"view": "active"}).json()
    resolved = client.get("/tickets", params={"view": "resolved"}).json()
    assert [t["id"] for t in active] == [ids[2]]
    assert sorted(t["id"] for t in resolved) == ids[:2]
    assert len(client.get("/tickets").json()) == 3
    assert client.get("/tickets", params={"view": "bogus"}).status_code == 422


def test_retriage_keeps_history(client):
    use_llm(None)
    tid = client.post("/tickets", json=TICKET).json()["id"]
    use_llm(FakeLLM(GOOD))
    t = client.post(f"/tickets/{tid}/retriage").json()
    assert [r["source"] for r in t["triage_results"]] == ["fallback", "llm"]
    assert t["summary"] == "Customer was charged twice for March."


def test_stats(client):
    use_llm(None)
    client.post("/tickets", json=TICKET)
    use_llm(FakeLLM(GOOD))
    client.post("/tickets", json=TICKET)
    s = client.get("/stats").json()
    assert s["total"] == 2 and s["fallback_rate"] == 0.5 and s["unassigned_open"] == 2


def test_404s(client):
    assert client.get("/tickets/999").status_code == 404
    assert client.post("/tickets/999/assign", json={"assignee": "x"}).status_code == 404


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Ticket Triage" in r.text