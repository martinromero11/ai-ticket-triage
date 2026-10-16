from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Category = Literal["billing", "technical", "account", "bug", "feature_request", "general"]
Priority = Literal["low", "medium", "high", "urgent"]
Status = Literal["open", "in_progress", "resolved", "closed"]

CATEGORIES: tuple[str, ...] = Category.__args__
PRIORITIES: tuple[str, ...] = Priority.__args__
STATUSES: tuple[str, ...] = Status.__args__


class TicketCreate(BaseModel):
    customer_email: str = Field(..., max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    subject: str = Field(..., min_length=1, max_length=255)
    body: str = Field(..., min_length=1, max_length=10_000)


class LLMTriage(BaseModel):
    """Schema the LLM output must satisfy before it is trusted."""

    summary: str = Field(..., min_length=5, max_length=500)
    category: Category
    priority: Priority
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("category", "priority", mode="before")
    @classmethod
    def normalize(cls, v):
        return v.strip().lower().replace(" ", "_") if isinstance(v, str) else v


class TriageResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    model: str | None
    summary: str
    category: str
    priority: str
    confidence: float | None
    error: str | None
    created_at: datetime


class StatusUpdateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    field: str
    old_value: str | None
    new_value: str | None
    note: str | None
    created_at: datetime


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    customer_email: str
    subject: str
    body: str
    summary: str | None
    category: str
    priority: str
    status: str
    assignee: str | None
    created_at: datetime
    updated_at: datetime


class TicketDetail(TicketOut):
    triage_results: list[TriageResultOut]
    status_updates: list[StatusUpdateOut]


class TicketUpdate(BaseModel):
    status: Status | None = None
    priority: Priority | None = None
    category: Category | None = None
    note: str | None = Field(None, max_length=1000)


class AssignRequest(BaseModel):
    assignee: str | None = Field(..., max_length=100)
    note: str | None = Field(None, max_length=1000)


class QueueStats(BaseModel):
    total: int
    by_status: dict[str, int]
    by_priority: dict[str, int]
    by_category: dict[str, int]
    unassigned_open: int
    fallback_rate: float
