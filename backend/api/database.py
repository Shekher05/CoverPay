"""Phase 5 - persistence.

SQLite for the MVP. The ORM code is identical for PostgreSQL, so moving over is
a change of `DATABASE_URL` in `.env` and nothing else - worth doing when the
Phase 7 simulator starts writing concurrently.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from config import settings

_is_sqlite = settings.database.startswith("sqlite")

# check_same_thread is a SQLite-only guard against sharing a connection across
# threads; FastAPI's threadpool needs it relaxed. `timeout` is how long a
# writer waits for the database lock before giving up - the default 5s is too
# short once the simulator writes while the API serves. Other backends ignore
# both.
connect_args = {"check_same_thread": False, "timeout": 30} if _is_sqlite else {}
engine = create_engine(settings.database, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):
        """WAL lets one writer and many readers proceed without blocking each
        other, which the default rollback journal does not. This is the MVP
        stopgap; the real fix is the documented move to PostgreSQL."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    """What the caller sent. Only the fields worth querying are columned out;
    the full payload is not stored, since it is the caller's own data."""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    amount: Mapped[float] = mapped_column(Float)
    transaction_dt: Mapped[int] = mapped_column(Integer, index=True)
    product_cd: Mapped[str | None] = mapped_column(String(8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Gateway entities. IEEE-CIS has none of these, so a transaction arriving
    # through the plain scoring API leaves them null; the simulator supplies them
    # and the behaviour and incident engines require them.
    customer_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    merchant_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    device_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    prediction: Mapped[Prediction] = relationship(back_populates="transaction")


class Prediction(Base):
    """What the model said, and why. Kept so a decision can be re-examined later
    against the model version that actually made it."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_pk: Mapped[int] = mapped_column(
        ForeignKey("transactions.id"), index=True
    )
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    recommendation: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(32))
    reasons: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # The other two engines' verdicts, when the transaction came through a
    # stream that had entity history to analyse. Null for a bare API call.
    behaviour_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    behaviour_detail: Mapped[str | None] = mapped_column(String(512), nullable=True)
    incident_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)

    transaction: Mapped[Transaction] = relationship(back_populates="prediction")


class IncidentRecord(Base):
    """A detected spike, as the incident engine left it.

    Stored flat rather than joined to its transactions: the membership link
    lives on `predictions.incident_id`, and this row is the incident's own
    lifecycle state.
    """

    __tablename__ = "incidents"

    incident_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16))
    severity: Mapped[str] = mapped_column(String(16))
    opened_dt: Mapped[int] = mapped_column(Integer, index=True)
    last_dt: Mapped[int] = mapped_column(Integer)
    transactions: Mapped[int] = mapped_column(Integer)
    customers: Mapped[int] = mapped_column(Integer)
    devices: Mapped[int] = mapped_column(Integer)
    total_value: Mapped[float] = mapped_column(Float)
    risky_value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def init_db() -> None:
    Base.metadata.create_all(engine)


def reset_db() -> None:
    """Drop and recreate. Used by the simulator feed, never by the API."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def get_session():
    """FastAPI dependency - one session per request, always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
