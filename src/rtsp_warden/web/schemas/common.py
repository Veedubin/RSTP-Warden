"""Common pagination and filter schemas."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PageParams(BaseModel):
    """Pagination parameters accepted in query strings."""

    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class PageResponse(BaseModel, Generic[T]):
    """Paginated response envelope."""

    items: list[T]
    total: int
    limit: int
    offset: int
