"""Twitch IRC parsing: IRCv3 tags, badges, bits, first-msg, replies, USERNOTICE, shared chat."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivtube.chat.twitch_irc import (
    MAX_MAPPED_DELAY_S,
    map_platform_ts,
    parse_irc,
    parse_irc_line,
    unescape_tag,
)
from aivtube.contracts.types import MsgKind, Platform
from aivtube.testing.fakes import SAMPLE_IRC_LINES

WALL = 1_760_000_020.0  # 20 s after the sample lines' base tmi-sent-ts
RECEIVED = 5000.0


def recorded_lines(fixtures_dir: Path) -> list[str]:
    text = (fixtures_dir / "irc" / "twitch_sample.txt").read_text(encoding="utf-8")
    return [line.replace("{channel}", "pailin_th") for line in text.splitlines() if line.strip()]


def by_id(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        parsed = parse_irc(line)
        assert parsed is not None
        out.setdefault(parsed.tags["id"][-4:], line)
    return out


def test_recorded_fixture_matches_the_fake_server_lines(fixtures_dir: Path) -> None:
    assert recorded_lines(fixtures_dir) == [
        line.replace("{channel}", "pailin_th") for line in SAMPLE_IRC_LINES
    ]


def test_parse_irc_structure() -> None:
    line = parse_irc("@a=1;b=x\\sy :nick!nick@nick.tmi.twitch.tv PRIVMSG #chan :hello : world\r\n")
    assert line is not None
    assert line.tags == {"a": "1", "b": "x y"}
    assert (line.prefix, line.nick, line.command) == (
        "nick!nick@nick.tmi.twitch.tv",
        "nick",
        "PRIVMSG",
    )
    assert line.params == ("#chan", "hello : world") and line.trailing == "hello : world"
    ping = parse_irc("PING :tmi.twitch.tv")
    assert ping is not None and (ping.command, ping.trailing) == ("PING", "tmi.twitch.tv")
    welcome = parse_irc(":tmi.twitch.tv 001 justinfan12345 :Welcome, GLHF!")
    assert welcome is not None and welcome.command == "001"
    assert welcome.params == ("justinfan12345", "Welcome, GLHF!")
    reconnect = parse_irc(":tmi.twitch.tv RECONNECT")
    assert reconnect is not None and (reconnect.command, reconnect.params) == ("RECONNECT", ())
    assert parse_irc("") is None and parse_irc("   \r\n") is None and parse_irc("@a=1") is None


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("plain", "plain"),
        ("a\\sb", "a b"),
        ("semi\\:colon", "semi;colon"),
        ("back\\\\slash", "back\\slash"),
        ("\\\\s", "\\s"),  # an escaped backslash followed by a literal s
        ("cr\\rlf\\n", "cr\rlf\n"),
        ("unknown\\q", "unknownq"),
        ("trailing\\", "trailing"),
    ],
)
def test_unescape_tag(raw: str, value: str) -> None:
    assert unescape_tag(raw) == value


def test_map_platform_ts() -> None:
    assert map_platform_ts(WALL - 1.5, received=RECEIVED, wall_now=WALL) == RECEIVED - 1.5
    assert map_platform_ts(WALL + 3.0, received=RECEIVED, wall_now=WALL) == RECEIVED  # skew
    far = WALL - MAX_MAPPED_DELAY_S - 1
    assert map_platform_ts(far, received=RECEIVED, wall_now=WALL) == RECEIVED
    assert map_platform_ts(None, received=RECEIVED, wall_now=WALL) == RECEIVED
    assert map_platform_ts(WALL - 1, received=RECEIVED, wall_now=None) == RECEIVED


def test_privmsg_badges_first_msg_and_timestamps(fixtures_dir: Path) -> None:
    lines = by_id(recorded_lines(fixtures_dir))

    first = parse_irc_line(lines["0001"], received=RECEIVED, wall_now=WALL)
    assert first is not None and first.platform is Platform.TWITCH
    assert first.id == "8c0f4d2e-0001-4000-8000-000000000001"
    assert (first.user.id, first.user.name) == ("1001", "Tom123")
    assert first.first_msg and first.kind is MsgKind.TEXT
    assert first.text == "สวัสดีครับไพลิน วันนี้เล่นเกมอะไร"
    assert first.ts == pytest.approx(RECEIVED - 20.0) and first.received == RECEIVED
    assert first.raw["room-id"] == "12345"

    sub = parse_irc_line(lines["0002"], received=RECEIVED, wall_now=WALL)
    assert sub is not None and sub.user.is_sub and sub.user.sub_months == 14
    assert not sub.first_msg and sub.user.name == "มะลิ"

    mod = parse_irc_line(lines["0003"], received=RECEIVED)
    assert mod is not None and mod.user.is_mod and not mod.user.is_sub
    assert mod.ts == RECEIVED  # no wall clock given

    vip = parse_irc_line(lines["0004"], received=RECEIVED)
    assert vip is not None and vip.user.is_vip and not vip.user.is_mod

    caster = parse_irc_line(lines["0013"], received=RECEIVED)
    assert caster is not None and caster.user.is_broadcaster


def test_bits_become_a_donation(fixtures_dir: Path) -> None:
    cheer = parse_irc_line(by_id(recorded_lines(fixtures_dir))["0005"], received=RECEIVED)
    assert cheer is not None and cheer.kind is MsgKind.DONATION
    assert (cheer.amount, cheer.currency, cheer.value_usd) == (100.0, "BITS", pytest.approx(1.0))
    assert cheer.text.startswith("Cheer100")


def test_usernotice_sub_resub_subgift_raid(fixtures_dir: Path) -> None:
    lines = by_id(recorded_lines(fixtures_dir))
    new_sub = parse_irc_line(lines["0007"], received=RECEIVED)
    assert new_sub is not None and new_sub.kind is MsgKind.SUB
    assert (new_sub.user.name, new_sub.user.id, new_sub.text) == ("NewSub", "2001", "")
    assert new_sub.user.is_sub and new_sub.value_usd == pytest.approx(4.99)
    assert new_sub.raw["system-msg"] == "NewSub subscribed at Tier 1."

    resub = parse_irc_line(lines["0008"], received=RECEIVED)
    assert resub is not None and resub.kind is MsgKind.SUB
    assert resub.user.sub_months == 6 and resub.text == "ครบหกเดือนแล้วนะไพลิน"

    gift = parse_irc_line(lines["0009"], received=RECEIVED)
    assert gift is not None and gift.kind is MsgKind.GIFT_SUB and gift.amount == 1.0
    assert gift.raw["msg-param-recipient-display-name"] == "LuckyViewer"

    raid = parse_irc_line(lines["0010"], received=RECEIVED)
    assert raid is not None and raid.kind is MsgKind.RAID
    assert (raid.amount, raid.user.name, raid.value_usd) == (42.0, "RaiderCh", 0.0)


def test_shared_chat_sets_source_channel(fixtures_dir: Path) -> None:
    shared = parse_irc_line(by_id(recorded_lines(fixtures_dir))["0011"], received=RECEIVED)
    assert shared is not None and shared.source_channel == "99999"
    same_room = parse_irc_line(
        "@id=x;room-id=12345;source-room-id=12345;user-id=1 :a!a@a PRIVMSG #pailin_th :hi",
        received=RECEIVED,
    )
    assert same_room is not None and same_room.source_channel is None


# Recorded-style lines for the tags the committed sample does not cover.
REAL_XQC = (
    "@badge-info=subscriber/19;badges=subscriber/2018,pichu/1;client-nonce=a0cc;color=#1E90FF;"
    "display-name=tojizen_in;emotes=;first-msg=0;flags=;id=8a21e66d-e149-4909-8857-be75f7fbe6d9;"
    "mod=0;returning-chatter=0;room-id=71092938;subscriber=1;tmi-sent-ts=1790319151556;turbo=0;"
    "user-id=452035063;user-type= :tojizen_in!tojizen_in@tojizen_in.tmi.twitch.tv PRIVMSG #xqc :by"
)
REPLY = (
    "@badge-info=;badges=;display-name=Replier;id=r-1;reply-parent-display-name=Tom123;"
    "reply-parent-msg-body=hi;reply-parent-msg-id=8c0f4d2e-0001-4000-8000-000000000001;"
    "reply-parent-user-id=1001;reply-parent-user-login=tom123;room-id=12345;"
    "tmi-sent-ts=1760000020000;user-id=1020 :replier!replier@replier.tmi.twitch.tv "
    "PRIVMSG #pailin_th :@Tom123 ใช่เลย"
)
ACTION = (
    "@badges=;display-name=;id=act-1;room-id=12345;tmi-sent-ts=1760000021000;user-id=1021 "
    ":waver!waver@waver.tmi.twitch.tv PRIVMSG #pailin_th :\x01ACTION โบกมือให้ไพลิน\x01"
)
FOUNDER = (
    "@badge-info=founder/35;badges=founder/0,partner/1;display-name=OG;id=f-1;room-id=12345;"
    "tmi-sent-ts=1760000022000;user-id=1022 :og!og@og.tmi.twitch.tv PRIVMSG #pailin_th :มาแล้ว"
)
REDEEM = (
    "@badges=;custom-reward-id=5d2b3c4e-0000-4000-8000-000000000abc;display-name=Redeemer;"
    "id=rd-1;room-id=12345;tmi-sent-ts=1760000023000;user-id=1023 "
    ":redeemer!redeemer@redeemer.tmi.twitch.tv PRIVMSG #pailin_th :ร้องเพลงให้ฟังหน่อย"
)
MYSTERY = (
    "@badges=;display-name=BigSanta;id=mg-1;login=bigsanta;msg-id=submysterygift;"
    "msg-param-mass-gift-count=5;msg-param-origin-id=xyz;msg-param-sender-count=50;"
    "msg-param-sub-plan=1000;room-id=12345;"
    "system-msg=BigSanta\\sis\\sgifting\\s5\\sTier\\s1\\sSubs!;tmi-sent-ts=1760000024000;"
    "user-id=2010 :tmi.twitch.tv USERNOTICE #pailin_th"
)
MYSTERY_PART = (
    "@badges=;display-name=BigSanta;id=mg-1a;login=bigsanta;msg-id=subgift;"
    "msg-param-community-gift-id=xyz;msg-param-recipient-display-name=A;msg-param-sub-plan=1000;"
    "room-id=12345;tmi-sent-ts=1760000024100;user-id=2010 :tmi.twitch.tv USERNOTICE #pailin_th"
)
ANON_GIFT = (
    "@badges=;display-name=AnAnonymousGifter;id=ag-1;login=ananonymousgifter;msg-id=anonsubgift;"
    "msg-param-recipient-display-name=B;msg-param-sub-plan=2000;room-id=12345;"
    "tmi-sent-ts=1760000025000;user-id=274598607 :tmi.twitch.tv USERNOTICE #pailin_th"
)
TIER3 = (
    "@badge-info=subscriber/1;badges=subscriber/3000;display-name=Rich;id=t3-1;login=rich;"
    "msg-id=sub;msg-param-cumulative-months=1;msg-param-sub-plan=3000;room-id=12345;"
    "tmi-sent-ts=1760000026000;user-id=2011 :tmi.twitch.tv USERNOTICE #pailin_th :ว้าว"
)
ANNOUNCEMENT = (
    "@badges=broadcaster/1;display-name=Streamer;id=an-1;login=streamer;msg-id=announcement;"
    "room-id=12345;tmi-sent-ts=1760000027000;user-id=12345 :tmi.twitch.tv USERNOTICE "
    "#pailin_th :ประกาศ"
)


def test_real_captured_line() -> None:
    m = parse_irc_line(REAL_XQC, received=RECEIVED)
    assert m is not None and m.user.is_sub and m.user.sub_months == 19 and m.text == "by"
    assert m.user.name == "tojizen_in" and m.user.id == "452035063"


def test_reply_action_founder_and_redeem() -> None:
    reply = parse_irc_line(REPLY, received=RECEIVED)
    assert reply is not None and reply.reply_to == "8c0f4d2e-0001-4000-8000-000000000001"
    action = parse_irc_line(ACTION, received=RECEIVED)
    assert action is not None and action.text == "โบกมือให้ไพลิน"
    assert action.user.name == "waver"  # empty display-name falls back to the login
    founder = parse_irc_line(FOUNDER, received=RECEIVED)
    assert founder is not None and founder.user.is_sub and founder.user.sub_months == 35
    assert founder.user.is_verified
    redeem = parse_irc_line(REDEEM, received=RECEIVED)
    assert redeem is not None and redeem.kind is MsgKind.REDEEM


def test_gift_batches_are_counted_once() -> None:
    mystery = parse_irc_line(MYSTERY, received=RECEIVED)
    assert mystery is not None and mystery.kind is MsgKind.GIFT_SUB
    assert mystery.amount == 5.0 and mystery.value_usd == pytest.approx(5 * 4.99)
    assert mystery.raw["system-msg"] == "BigSanta is gifting 5 Tier 1 Subs!"
    assert parse_irc_line(MYSTERY_PART, received=RECEIVED) is None
    anon = parse_irc_line(ANON_GIFT, received=RECEIVED)
    assert anon is not None and anon.kind is MsgKind.GIFT_SUB
    assert anon.value_usd == pytest.approx(9.99)
    tier3 = parse_irc_line(TIER3, received=RECEIVED)
    assert tier3 is not None and tier3.value_usd == pytest.approx(24.99) and tier3.text == "ว้าว"


@pytest.mark.parametrize(
    "line",
    [
        ANNOUNCEMENT,
        "PING :tmi.twitch.tv",
        ":tmi.twitch.tv RECONNECT",
        "@room-id=12345 :tmi.twitch.tv ROOMSTATE #pailin_th",
        "@room-id=12345;target-user-id=1 :tmi.twitch.tv CLEARCHAT #pailin_th :baduser",
        ":tmi.twitch.tv NOTICE * :Login authentication failed",
        "@id=x :a!a@a PRIVMSG #pailin_th",  # no text parameter
        "garbage",
    ],
)
def test_non_chat_lines_are_not_messages(line: str) -> None:
    assert parse_irc_line(line, received=RECEIVED) is None


def test_missing_id_gets_a_stable_fallback() -> None:
    line = "@room-id=1;tmi-sent-ts=5;user-id=9 :a!a@a PRIVMSG #c :hi"
    a = parse_irc_line(line, received=1.0)
    b = parse_irc_line(line, received=2.0)
    assert a is not None and b is not None and a.id == b.id == "1:9:5"
