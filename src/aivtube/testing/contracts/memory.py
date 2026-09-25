"""Contract suite for ``MemoryStore`` (§3.9, §6)."""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from aivtube.contracts.memory import MemoryItem, MemoryStore, PrefixMemory, SlotsFull, Turn
from aivtube.testing.contracts._base import (
    AsyncCase,
    ContractViolation,
    _Cases,
    check,
    close_quietly,
    maybe_await,
)

__all__ = ["memory_store_suite"]


def memory_store_suite(
    factory: Callable[[], MemoryStore | Awaitable[MemoryStore]],
    *,
    core_slots: int = 16,
) -> list[AsyncCase]:
    """``factory`` returns a fresh, empty store (for SQLite: a new file in a temp dir)."""
    cases = _Cases("memory_store")

    async def fresh() -> MemoryStore:
        store = await maybe_await(factory())
        await store.start_session("contract")
        return store

    def core(text: str, **kw: object) -> MemoryItem:
        return MemoryItem(id=None, kind="core", text=text, source="operator", **kw)  # type: ignore[arg-type]

    @cases
    async def turns_round_trip_in_order() -> None:
        store = await fresh()
        try:
            prefix = await store.prefix_block()
            ids = [
                await store.append_turn(Turn(role="user", text="สวัสดีไพลิน", source="voice")),
                await store.append_turn(Turn(role="assistant", text="สวัสดีค่ะ", source="speak")),
            ]
            check(ids == sorted(ids) and len(set(ids)) == 2, f"turn ids {ids}")
            turns = await store.recent_turns(prefix.epoch)
            check([t.text for t in turns][-2:] == ["สวัสดีไพลิน", "สวัสดีค่ะ"], "turn order/text")
            check(all(t.id is not None for t in turns), "stored turns must carry their id")
        finally:
            await close_quietly(store)

    @cases
    async def remember_assigns_a_core_slot() -> None:
        store = await fresh()
        try:
            item = await store.remember(core("ไพลินชอบแมวสีส้ม"))
            check(item.id is not None, "remember() must return the stored id")
            check(item.slot is not None and 1 <= item.slot <= core_slots, f"slot {item.slot}")
            listed = await store.list_memories(kind="core")
            check(any(m.id == item.id for m in listed), "list_memories misses the new item")
        finally:
            await close_quietly(store)

    @cases
    async def slots_full_then_replace_slot() -> None:
        store = await fresh()
        try:
            items = [await store.remember(core(f"ความจำหมายเลข {i}")) for i in range(core_slots)]
            try:
                await store.remember(core("ความจำที่เกินมา"))
            except SlotsFull as exc:
                check(len(exc.slots) == core_slots, f"SlotsFull.slots has {len(exc.slots)} items")
            else:
                raise ContractViolation(f"remember #{core_slots + 1} did not raise SlotsFull")
            slot = items[3].slot
            new = await store.remember(core("ความจำใหม่แทนที่"), replace_slot=slot)
            check(new.slot == slot, f"replace_slot put it in {new.slot}")
            active = {m.text for m in await store.list_memories(kind="core", status="active")}
            check("ความจำใหม่แทนที่" in active and items[3].text not in active, "replace failed")
        finally:
            await close_quietly(store)

    @cases
    async def locked_items_refuse_forget() -> None:
        store = await fresh()
        try:
            locked = await store.remember(core("ชื่อของไพลินคือไพลิน", locked=True))
            assert locked.id is not None
            with contextlib.suppress(Exception):
                await store.forget(locked.id, by="contract", reason="test")
            active = {m.id for m in await store.list_memories(status="active")}
            check(locked.id in active, "a locked item was forgotten")
            free = await store.remember(core("ลืมได้"))
            assert free.id is not None
            await store.forget(free.id, by="contract", reason="test")
            active = {m.id for m in await store.list_memories(status="active")}
            check(free.id not in active, "forget() did not remove the item")
        finally:
            await close_quietly(store)

    @cases
    async def quarantined_items_stay_out_of_the_prefix() -> None:
        store = await fresh()
        try:
            q = await store.remember(core("ข้อมูลจากแชทที่ต้องตรวจก่อน", status="quarantined"))
            assert q.id is not None
            prefix = await store.prefix_block()
            check(isinstance(prefix, PrefixMemory), "prefix_block() type")
            check(q.id not in {m.id for m in prefix.core}, "quarantined item in the prefix")
            await store.set_status(q.id, "active", by="operator")
            prefix = await store.prefix_block()
            check(q.id in {m.id for m in prefix.core}, "approved item missing from the prefix")
        finally:
            await close_quietly(store)

    @cases
    async def epochs_and_digest_stability() -> None:
        store = await fresh()
        try:
            before = await store.prefix_block()
            t = await store.append_turn(Turn(role="user", text="เริ่ม", source="voice"))
            epoch = await store.new_epoch("", t, "hash-a")
            check(epoch > before.epoch, "new_epoch() must increase")
            same = await store.prefix_block()
            check(same.epoch == epoch, "prefix_block().epoch")
            check(same.digest == before.digest, "digest changed without a content change")
            await store.remember(core("ความจำใหม่ในสล็อต"))
            changed = await store.prefix_block()
            check(changed.digest != same.digest, "digest did not change with the core slots")
            epoch2 = await store.new_epoch("สรุป: คุยเรื่องเกม", t, "hash-b")
            summarised = await store.prefix_block()
            check(summarised.rolling_summary == "สรุป: คุยเรื่องเกม", "rolling summary")
            check(summarised.epoch == epoch2 and summarised.digest != changed.digest, "digest")
        finally:
            await close_quietly(store)

    @cases
    async def pending_since_epoch_lists_new_items() -> None:
        store = await fresh()
        try:
            old = await store.remember(core("ก่อนรอบใหม่"))
            t = await store.append_turn(Turn(role="user", text="x", source="voice"))
            await store.new_epoch("", t, "h")
            new = await store.remember(core("หลังรอบใหม่"))
            pending = {m.id for m in await store.pending_since_epoch()}
            check(new.id in pending and old.id not in pending, f"pending {pending}")
        finally:
            await close_quietly(store)

    @cases
    async def search_finds_thai_substrings() -> None:
        store = await fresh()
        try:
            await store.remember(
                MemoryItem(id=None, kind="fact", text="ไพลินชอบกินข้าวมันไก่มาก", source="operator")
            )
            hits = await store.search("มันไก่")
            check(any("มันไก่" in m.text for m in hits), "Thai search missed 'มันไก่'")
            check(len(await store.search("มันไก่", k=1)) <= 1, "k is not respected")
        finally:
            await close_quietly(store)

    @cases
    async def viewer_facts_only_active() -> None:
        store = await fresh()
        try:
            await store.upsert_viewer("twitch", "u-7", "มะลิ")
            active = await store.remember(
                MemoryItem(
                    id=None, kind="viewer", text="มะลิชอบเล่นมายคราฟ", platform="twitch",
                    user_id="u-7", source="operator",
                )
            )  # fmt: skip
            await store.remember(
                MemoryItem(
                    id=None, kind="viewer", text="ยังไม่ได้ตรวจ", platform="twitch",
                    user_id="u-7", source="model", status="quarantined",
                )
            )  # fmt: skip
            facts = await store.viewer_facts([("twitch", "u-7")])
            check([m.id for m in facts] == [active.id], f"viewer facts {facts!r}")
            check(await store.viewer_facts([("twitch", "nobody")]) == [], "facts for a stranger")
        finally:
            await close_quietly(store)

    @cases
    async def session_resume_and_end() -> None:
        store = await maybe_await(factory())
        try:
            sid = await store.start_session("resume")
            check(await store.resume_session(3600.0) == sid, "a fresh session must resume")
            await store.end_session("จบสตรีม")
            check(await store.resume_session(3600.0) is None, "an ended session resumed")
        finally:
            await close_quietly(store)

    @cases
    async def backup_writes_a_file() -> None:
        store = await fresh()
        try:
            await store.remember(core("สำรองข้อมูล"))
            with tempfile.TemporaryDirectory() as d:
                path = await store.backup(Path(d), keep=2)
                check(isinstance(path, Path) and path.exists(), "backup() must return a file")
        finally:
            await close_quietly(store)

    return cases.items
