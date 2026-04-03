"""
User context models — persistent, cross-session state keyed by browser fingerprint.

Stored in Redis with no TTL so the system "remembers" users across sessions.
Conversation turns remain session-scoped (ephemeral); UserContext is permanent.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class Preferences(BaseModel):
    """User preferences — overwrite semantics on update."""

    name: str | None = None
    default_location: str | None = None
    temperature_unit: str = "fahrenheit"  # "fahrenheit" | "celsius"
    language: str = "en"
    watched_tickers: list[str] = Field(default_factory=list)


class MemoryFact(BaseModel):
    """A single inferred fact about the user."""

    fact: str
    source_domain: str
    timestamp: float = Field(default_factory=time.time)


class MemoryStore(BaseModel):
    """Thinker-written observations about the user, capped at MAX_FACTS."""

    MAX_FACTS: int = 20
    facts: list[MemoryFact] = Field(default_factory=list)

    def add_fact(self, fact: str, source_domain: str) -> None:
        """Append a fact, deduplicating by text and capping at MAX_FACTS."""
        normalised = fact.strip().lower()
        if any(f.fact.strip().lower() == normalised for f in self.facts):
            return
        self.facts.append(MemoryFact(fact=fact, source_domain=source_domain))
        if len(self.facts) > self.MAX_FACTS:
            self.facts = self.facts[-self.MAX_FACTS :]


class Summary(BaseModel):
    """Rolling conversation summary, regenerated periodically."""

    text: str = ""
    turn_count_at_summary: int = 0
    updated_at: float = 0.0


class Signals(BaseModel):
    """Behavioural analytics signals."""

    topic_counts: dict[str, int] = Field(default_factory=dict)
    last_active: float = Field(default_factory=time.time)
    session_count: int = 0

    def record_topic(self, domain: str) -> None:
        self.topic_counts[domain] = self.topic_counts.get(domain, 0) + 1
        self.last_active = time.time()


class UserContext(BaseModel):
    """Top-level user context object persisted in Redis."""

    preferences: Preferences = Field(default_factory=Preferences)
    memory: MemoryStore = Field(default_factory=MemoryStore)
    summary: Summary = Field(default_factory=Summary)
    signals: Signals = Field(default_factory=Signals)


# ── Thinker output types ────────────────────────────────────────────────


class ContextUpdate(BaseModel):
    """Incremental updates a Thinker wants to apply to UserContext."""

    # Preferences (overwrite semantics — only set fields are applied)
    set_name: str | None = None
    set_default_location: str | None = None
    set_temperature_unit: str | None = None
    add_watched_tickers: list[str] = Field(default_factory=list)

    # Memory (append semantics)
    new_facts: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return self.set_name is None and self.set_default_location is None and self.set_temperature_unit is None and not self.add_watched_tickers and not self.new_facts


class ThinkResult(BaseModel):
    """Return type for BaseThinker.think() — response + optional context updates."""

    response: str
    context_update: ContextUpdate | None = None
