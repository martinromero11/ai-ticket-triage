import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session, selectinload

from . import models
from .database import Base, engine, get_db
from .schemas import (
    AssignRequest,
    Category,
    Priority,
    QueueStats,
    Status,
    TicketCreate,
    TicketDetail,
    TicketOut,
    TicketUpdate,
)
from .triage import LLMClient, TriageOutcome, default_client, triage_ticket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Support Ticket Triage", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"

VIEW_STATUSES = {
    "active": ("open", "in_progress"),
    "resolved": ("resolved", "closed"),
}

PRIORITY_RANK = case(
    {"urgent": 0, "high": 1, "medium": 2, "low": 3}, value=models.Ticket.priority, else_=4
)


def get_llm_client() -> LLMClient | None:
    return default_client()


def _get_ticket(db: Session, ticket_id: int) -> models.Ticket:
    ticket = db.get(models.Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


def _apply_triage(db: Session, ticket: models.Ticket, outcome: TriageOutcome) -> None:
    db.add(
        models.TriageResult(
            ticket=ticket,
            source=outcome.source,
            model=outcome.model,
            summary=outcome.summary,
            category=outcome.category,
            priority=outcome.priority,
            confidence=outcome.confidence,
            raw_output=outcome.raw_output,
            error=outcome.error,
        )
    )
    ticket.summary = outcome.summary
    ticket.category = outcome.category
    ticket.priority = outcome.priority


def _record_change(db, ticket, field, new_value, note=None) -> bool:
    old_value = getattr(ticket, field)
    if old_value == new_value:
        return False
    setattr(ticket, field, new_value)
    db.add(
        models.StatusUpdate(
            ticket=ticket, field=field, old_value=old_value, new_value=new_value, note=note
        )
    )
    return True


# ---------- dashboard & health ----------


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/health")
def health(llm: LLMClient | None = Depends(get_llm_client)):
    return {"status": "ok", "llm_configured": llm is not None, "model": getattr(llm, "model", None)}


# ---------- tickets ----------


@app.post("/tickets", response_model=TicketDetail, status_code=201)
def create_ticket(
    payload: TicketCreate,
    db: Session = Depends(get_db),
    llm: LLMClient | None = Depends(get_llm_client),
):
    ticket = models.Ticket(**payload.model_dump())
    db.add(ticket)
    _apply_triage(db, ticket, triage_ticket(payload.subject, payload.body, llm))
    db.commit()
    db.refresh(ticket)
    log.info("ticket %s created: %s/%s", ticket.id, ticket.category, ticket.priority)
    return ticket


@app.get("/tickets", response_model=list[TicketOut])
def list_tickets(
    view: str = Query("all", pattern="^(active|resolved|all)$",
                      description="active = open + in_progress, resolved = resolved + closed"),
    status: Status | None = None,
    priority: Priority | None = None,
    category: Category | None = None,
    assignee: str | None = None,
    unassigned: bool = False,
    q: str | None = Query(None, description="Search subject, body and summary"),
    sort: str = Query("newest", pattern="^(newest|oldest|priority)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(models.Ticket)
    if view in VIEW_STATUSES:
        stmt = stmt.where(models.Ticket.status.in_(VIEW_STATUSES[view]))
    if status:
        stmt = stmt.where(models.Ticket.status == status)
    if priority:
        stmt = stmt.where(models.Ticket.priority == priority)
    if category:
        stmt = stmt.where(models.Ticket.category == category)
    if assignee:
        stmt = stmt.where(models.Ticket.assignee == assignee)
    if unassigned:
        stmt = stmt.where(models.Ticket.assignee.is_(None))
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(
                models.Ticket.subject.ilike(like),
                models.Ticket.body.ilike(like),
                models.Ticket.summary.ilike(like),
            )
        )

    order = {
        "newest": [models.Ticket.id.desc()],
        "oldest": [models.Ticket.id.asc()],
        "priority": [PRIORITY_RANK, models.Ticket.id.asc()],
    }[sort]
    return db.scalars(stmt.order_by(*order).limit(limit).offset(offset)).all()


@app.get("/tickets/{ticket_id}", response_model=TicketDetail)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    stmt = (
        select(models.Ticket)
        .where(models.Ticket.id == ticket_id)
        .options(
            selectinload(models.Ticket.triage_results),
            selectinload(models.Ticket.status_updates),
        )
    )
    ticket = db.scalars(stmt).first()
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.patch("/tickets/{ticket_id}", response_model=TicketDetail)
def update_ticket(ticket_id: int, payload: TicketUpdate, db: Session = Depends(get_db)):
    ticket = _get_ticket(db, ticket_id)
    changes = payload.model_dump(exclude_unset=True, exclude={"note"})
    if not changes:
        raise HTTPException(status_code=422, detail="No fields to update")
    for field, value in changes.items():
        if value is not None:
            _record_change(db, ticket, field, value, payload.note)
    db.commit()
    db.refresh(ticket)
    return ticket


@app.post("/tickets/{ticket_id}/assign", response_model=TicketDetail)
def assign_ticket(ticket_id: int, payload: AssignRequest, db: Session = Depends(get_db)):
    ticket = _get_ticket(db, ticket_id)
    assignee = payload.assignee.strip() if payload.assignee else None
    _record_change(db, ticket, "assignee", assignee or None, payload.note)
    if assignee and ticket.status == "open":
        _record_change(db, ticket, "status", "in_progress", "auto: assigned")
    db.commit()
    db.refresh(ticket)
    return ticket


@app.post("/tickets/{ticket_id}/retriage", response_model=TicketDetail)
def retriage_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
    llm: LLMClient | None = Depends(get_llm_client),
):
    ticket = _get_ticket(db, ticket_id)
    _apply_triage(db, ticket, triage_ticket(ticket.subject, ticket.body, llm))
    db.commit()
    db.refresh(ticket)
    return ticket


@app.delete("/tickets/{ticket_id}", status_code=204)
def delete_ticket(ticket_id: int, db: Session = Depends(get_db)):
    db.delete(_get_ticket(db, ticket_id))
    db.commit()


# ---------- queue ----------


@app.get("/queue", response_model=list[TicketOut])
def work_queue(
    assignee: str | None = None,
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Active tickets ordered by priority, then oldest first."""
    stmt = select(models.Ticket).where(models.Ticket.status.in_(VIEW_STATUSES["active"]))
    if assignee:
        stmt = stmt.where(models.Ticket.assignee == assignee)
    stmt = stmt.order_by(PRIORITY_RANK, models.Ticket.id.asc()).limit(limit)
    return db.scalars(stmt).all()


@app.get("/stats", response_model=QueueStats)
def queue_stats(db: Session = Depends(get_db)):
    def counts(col):
        return dict(db.execute(select(col, func.count()).group_by(col)).all())

    total = db.scalar(select(func.count()).select_from(models.Ticket)) or 0
    unassigned_open = db.scalar(
        select(func.count())
        .select_from(models.Ticket)
        .where(models.Ticket.assignee.is_(None), models.Ticket.status == "open")
    ) or 0
    runs = db.scalar(select(func.count()).select_from(models.TriageResult)) or 0
    fallbacks = db.scalar(
        select(func.count())
        .select_from(models.TriageResult)
        .where(models.TriageResult.source == "fallback")
    ) or 0
    return QueueStats(
        total=total,
        by_status=counts(models.Ticket.status),
        by_priority=counts(models.Ticket.priority),
        by_category=counts(models.Ticket.category),
        unassigned_open=unassigned_open,
        fallback_rate=round(fallbacks / runs, 3) if runs else 0.0,
    )