"""Helpers shared by the contract suites (no pytest dependency)."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, TypeVar

__all__ = [
    "AsyncCase",
    "Case",
    "SyncCase",
    "await_until",
    "case_id",
    "maybe_await",
    "raises",
    "wait_until",
]

T = TypeVar("T")
SyncCase = Callable[[], None]
AsyncCase = Callable[[], Awaitable[None]]
Case = SyncCase | AsyncCase


class ContractViolation(AssertionError):
    """A contract suite check failed."""


def case_id(case: Callable[..., Any]) -> str:
    """Readable pytest id for a suite case (``ids=case_id``)."""
    return getattr(case, "__name__", repr(case))


class _Cases:
    """Collects the case functions of one suite and names them ``<suite>.<case>``."""

    def __init__(self, suite: str) -> None:
        self.suite = suite
        self.items: list[Any] = []

    def __call__(self, fn: Callable[[], T]) -> Callable[[], T]:
        fn.__name__ = f"{self.suite}.{fn.__name__}"
        fn.__qualname__ = fn.__name__
        self.items.append(fn)
        return fn


@contextlib.contextmanager
def raises(*types: type[BaseException], what: str = "") -> Iterator[None]:
    """Like ``pytest.raises`` without pytest: the block must raise one of ``types``."""
    expected = types or (Exception,)
    try:
        yield
    except expected:
        return
    raise ContractViolation(
        f"expected {', '.join(t.__name__ for t in expected)}: {what}".rstrip(": ")
    )


def wait_until(predicate: Callable[[], bool], timeout: float = 2.0, *, what: str = "") -> None:
    """Poll ``predicate`` in real time (for callbacks fired from other threads)."""
    end = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > end:
            raise ContractViolation(f"timed out waiting for {what or 'condition'}")
        time.sleep(0.005)


async def await_until(
    predicate: Callable[[], bool],
    *,
    within: float = 5.0,
    advance: Callable[[float], Awaitable[None]] | None = None,
    step: float = 0.02,
    what: str = "",
) -> None:
    """Wait until ``predicate()``; with ``advance`` the harness drives (fake) time in steps."""
    end = time.perf_counter() + within
    waited = 0.0
    while not predicate():
        if advance is not None:
            if waited > within:
                raise ContractViolation(f"timed out waiting for {what or 'condition'} (fake time)")
            await advance(step)
            waited += step
        else:
            if time.perf_counter() > end:
                raise ContractViolation(f"timed out waiting for {what or 'condition'}")
            await asyncio.sleep(0.005)


async def maybe_await(value: T | Awaitable[T]) -> T:
    """Factories may be sync or async; this accepts both."""
    if inspect.isawaitable(value):
        return await value
    return value


async def close_quietly(obj: object) -> None:
    """Best-effort ``aclose()``/``close()`` of whatever a factory returned."""
    for name in ("aclose", "close", "stop"):
        fn = getattr(obj, name, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                res = fn()
                if inspect.isawaitable(res):
                    await res
            return


def check(cond: object, message: str) -> None:
    """``assert`` that survives ``python -O``."""
    if not cond:
        raise ContractViolation(message)
