"""Typed configuration (ARCHITECTURE.md §8): ``AppConfig``, ``CharacterConfig``, ``Secrets``.

Every model forbids unknown keys, so typos surface as errors. Models are frozen: build a
changed copy with ``model_copy(update=...)`` or reload through ``load_config``.

The field defaults mirror ``config/defaults.toml`` (a unit test enforces this), so
``AppConfig()`` is a complete, valid stream-profile config that needs no files.

Consent (invariant I8): after validation, every cloud LLM provider is ``enabled=False`` unless
``privacy.cloud_llm_consent`` is true, and every cloud STT backend unless
``privacy.cloud_stt_consent`` is true. Missing consent disables; it is never an error.
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict

from aivtube.config.layers import CHARACTER_PLACEHOLDER, CURRENT_SCHEMA_VERSION, expand_character
from aivtube.contracts.speech import BargePolicy, EchoMode, MicMode
from aivtube.contracts.types import Platform, VoiceSpec

if TYPE_CHECKING:
    from aivtube.contracts.voice import EndpointerConfig

__all__ = [
    "VTS_PORT_RANGE",
    "AppConfig",
    "AppSection",
    "AudioConfig",
    "AvatarConfig",
    "BargeInConfig",
    "BrainBudget",
    "BrainConfig",
    "CharacterAvatarConfig",
    "CharacterConfig",
    "CharacterGamesConfig",
    "CharacterMemoryConfig",
    "CharacterSafetyConfig",
    "CharacterToolsConfig",
    "ChatConfig",
    "ChatWindowConfig",
    "ChunkerSection",
    "EmotionPose",
    "GamesConfig",
    "LauncherConfig",
    "LlamaServerConfig",
    "LlmConfig",
    "LlmProviderConfig",
    "LlmSlotsConfig",
    "LoggingConfig",
    "MemoryConfig",
    "MicConfig",
    "PanelConfig",
    "PortsConfig",
    "PrivacyConfig",
    "SafetyConfig",
    "Secrets",
    "SttBackendConfig",
    "SttConfig",
    "ToolsConfig",
    "TtsBackendConfig",
    "TtsConfig",
    "TtsIdentityConfig",
    "TwitchIrcConfig",
    "VadConfig",
    "YouTubePollConfig",
    "is_secret_name",
    "is_valid_character_id",
    "load_lexicon",
]

VTS_PORT_RANGE = range(8001, 8010)  # VTube Studio instances; we never bind these
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
EMOTIONS = ("neutral", "happy", "sad", "angry", "surprised", "shy", "smug")
_CHAR_ID = re.compile(r"^[a-z_][a-z0-9_]{0,31}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PLUGIN_KIND = re.compile(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


# --- helpers ----------------------------------------------------------------------------


def fail(
    code: str,
    en: str,
    th: str,
    hint: str = "",
    hint_th: str = "",
    *,
    field: str | None = None,
    path: str | None = None,
    also: str | None = None,
) -> PydanticCustomError:
    """A validation error that carries Thai text and a hint for ``ConfigError``.

    ``field`` is appended to the error location; ``path`` replaces it (absolute dotted key).
    ``also`` names a second key involved (a clash): the error is reported at whichever of the
    two was set by the higher config layer.
    """
    ctx: dict[str, Any] = {"en": en, "th": th, "hint": hint, "hint_th": hint_th}
    if field:
        ctx["field"] = field
    if path:
        ctx["path"] = path
    if also:
        ctx["also"] = also
    return PydanticCustomError(code, "{en}", ctx)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    def _changed(self, name: str) -> bool:
        """True when ``name`` differs from its declared default."""
        info = type(self).model_fields[name]
        default = info.get_default(call_default_factory=True)
        return bool(getattr(self, name) != default)


def _check_port(value: int) -> int:
    if not 1 <= value <= 65535:
        raise fail(
            "port_range",
            f"Port {value} is outside 1–65535.",
            f"พอร์ต {value} อยู่นอกช่วง 1–65535",
            "Pick a free port such as 8770–8799.",
            "เลือกพอร์ตว่าง เช่น 8770–8799",
        )
    if value in VTS_PORT_RANGE:
        raise fail(
            "port_vts_reserved",
            f"Port {value} is reserved for VTube Studio (8001–8009).",
            f"พอร์ต {value} สงวนไว้ให้ VTube Studio (8001–8009)",
            "Use another port; the twin's SDK hub uses 8010.",
            "ใช้พอร์ตอื่น (ช่อง SDK ของตัวละครที่สองใช้ 8010)",
        )
    return value


Port = Annotated[int, AfterValidator(_check_port)]


def _url_port(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None and parts.scheme in ("http", "ws"):
        port = 80
    elif port is None and parts.scheme in ("https", "wss"):
        port = 443
    return parts.scheme, (parts.hostname or ""), port


def _is_loopback(url: str) -> bool:
    return _url_port(url)[1] in LOOPBACK_HOSTS


def _check_http_url(url: str, field: str) -> None:
    scheme, host, _ = _url_port(url)
    if scheme not in ("http", "https") or not host:
        raise fail(
            "url",
            f"{field} must be an http(s) URL, got {url!r}.",
            f"{field} ต้องเป็น URL แบบ http(s) แต่ได้ {url!r}",
            "For example: http://127.0.0.1:8080/v1",
            "ตัวอย่าง: http://127.0.0.1:8080/v1",
            field=field,
        )


def _check_env_name(value: str | None, field: str) -> None:
    if value is not None and not _ENV_NAME.match(value):
        raise fail(
            "env_name",
            f"{field} must name an environment variable (UPPER_CASE), got {value!r}.",
            f"{field} ต้องเป็นชื่อตัวแปรใน .env (ตัวพิมพ์ใหญ่) แต่ได้ {value!r}",
            "Put the secret itself in .env and reference its name here.",
            "ใส่ค่าลับไว้ในไฟล์ .env แล้วอ้างถึงชื่อตัวแปรที่นี่",
            field=field,
        )


def _check_kind(kind: str, known: Mapping[str, frozenset[str]], section: str) -> None:
    if kind in known or _PLUGIN_KIND.match(kind):
        return
    names = ", ".join(sorted(known))
    raise fail(
        "unknown_kind",
        f"Unknown {section} kind {kind!r}. Built-in kinds: {names}.",
        f"ไม่รู้จักชนิด {section} {kind!r} ชนิดที่มี: {names}",
        "Plugins use a namespaced kind such as 'mypkg:name'.",
        "ปลั๊กอินต้องใช้ชื่อแบบมีเนมสเปซ เช่น 'mypkg:name'",
        field="kind",
    )


def _check_kind_fields(model: _Model, kind: str, known: Mapping[str, frozenset[str]]) -> None:
    allowed = known.get(kind)
    if allowed is None:  # plugin kind: any field may be used
        return
    common = {"kind", "enabled", "cloud"}
    for name in type(model).model_fields:
        if name not in common and name not in allowed and model._changed(name):
            raise fail(
                "field_not_for_kind",
                f"{name!r} does not apply to kind {kind!r}.",
                f"คีย์ {name!r} ใช้กับชนิด {kind!r} ไม่ได้",
                f"Remove {name!r} (valid here: {', '.join(sorted(allowed)) or 'none'}).",
                f"ลบ {name!r} ออก",
                field=name,
            )


def _check_device(value: Any) -> Any:
    hint = 'Use (part of) the device name, e.g. "Headphones"; `aivtube setup --audio` lists them.'
    hint_th = 'ใช้ชื่ออุปกรณ์ (หรือบางส่วนของชื่อ) เช่น "Headphones" ดูรายชื่อได้ด้วย aivtube setup --audio'
    if isinstance(value, bool | int | float) or (
        isinstance(value, str) and value.strip().isdigit()
    ):
        raise fail(
            "device_index",
            f"Audio devices are chosen by name, not by number ({value!r}); numbers change when "
            "devices are plugged in.",
            f"ต้องเลือกอุปกรณ์เสียงด้วยชื่อ ไม่ใช่หมายเลข ({value!r}) เพราะหมายเลขเปลี่ยนได้เมื่อเสียบหรือถอดอุปกรณ์",
            hint,
            hint_th,
        )
    if not isinstance(value, str):
        return value
    if value and not value.strip():
        raise fail(
            "device_blank",
            'Device name is only whitespace; use "" for the Windows default device.',
            'ชื่ออุปกรณ์มีแต่ช่องว่าง ถ้าจะใช้อุปกรณ์ค่าเริ่มต้นของ Windows ให้ใส่ ""',
            hint,
            hint_th,
        )
    if any(unicodedata.category(ch).startswith("C") for ch in value) or len(value) > 200:
        raise fail(
            "device_name",
            f"Invalid device name {value[:40]!r} (control characters or too long).",
            f"ชื่ออุปกรณ์ {value[:40]!r} ไม่ถูกต้อง (มีอักขระควบคุมหรือยาวเกินไป)",
            hint,
            hint_th,
        )
    return value.strip()


DeviceName = Annotated[str, BeforeValidator(_check_device)]


def is_valid_character_id(value: str) -> bool:
    """Character ids: lowercase letters, digits and ``_``; at most 32 characters."""
    return bool(_CHAR_ID.match(value))


def _check_char_id(value: str) -> str:
    if not is_valid_character_id(value):
        raise fail(
            "character_id",
            f"Invalid character id {value!r}: use lowercase letters, digits and _ (max 32).",
            f"รหัสตัวละคร {value!r} ไม่ถูกต้อง ใช้ได้เฉพาะตัวพิมพ์เล็ก ตัวเลข และ _ (ไม่เกิน 32 ตัว)",
            "The id must match the folder name under characters/.",
            "รหัสต้องตรงกับชื่อโฟลเดอร์ใน characters/",
        )
    return value


CharId = Annotated[str, AfterValidator(_check_char_id)]


def _check_name(value: str) -> str:
    if not _NAME.match(value):
        raise fail(
            "name",
            f"Invalid name {value!r}: use lowercase letters, digits, '-', '_' or '.'.",
            f"ชื่อ {value!r} ไม่ถูกต้อง ใช้ได้เฉพาะตัวพิมพ์เล็ก ตัวเลข '-', '_' หรือ '.'",
        )
    return value


Name = Annotated[str, AfterValidator(_check_name)]


def _check_backoff(value: list[float]) -> list[float]:
    if len(value) != 2 or not 0 < value[0] <= value[1]:
        raise fail(
            "backoff",
            f"Backoff must be [min, max] with 0 < min <= max, got {value!r}.",
            f"ค่า backoff ต้องเป็น [ต่ำสุด, สูงสุด] โดย 0 < ต่ำสุด <= สูงสุด แต่ได้ {value!r}",
            "For example: [0.5, 30.0]",
            "ตัวอย่าง: [0.5, 30.0]",
        )
    return value


Backoff = Annotated[list[float], AfterValidator(_check_backoff)]


def _unique(values: list[str], field: str) -> None:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise fail(
                "duplicate",
                f"{v!r} is listed twice in {field}.",
                f"{v!r} ซ้ำกันใน {field}",
                "Remove the duplicate.",
                "ลบรายการที่ซ้ำ",
                field=field,
            )
        seen.add(v)


# --- sections ---------------------------------------------------------------------------


class AppSection(_Model):
    data_dir: str = "data"
    models_dir: str = "models"
    voice_worker: bool = True
    fakes: bool = False
    auto_live: bool = False
    resume_live_after_crash: bool = True
    session_resume_max_age_s: float = Field(default=21600.0, ge=0)
    monetised: bool = False


class PortsConfig(_Model):
    panel: Port = 8770
    bus: Port = 8771
    emergency: Port = 8779
    neuro_sdk: Port = 8000
    browser_renderer_http: Port = 8765
    browser_renderer_ws: Port = 8766


class PrivacyConfig(_Model):
    cloud_llm_consent: bool = False
    cloud_stt_consent: bool = False
    log_chat_text: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)


class AudioConfig(_Model):
    output_device: DeviceName = ""
    input_device: DeviceName = ""
    mirror_output_device: DeviceName = ""
    headphones: bool = True
    echo_mode: EchoMode = "auto"
    samplerate: Literal[44100, 48000] = 48000
    block_ms: int = Field(default=10, ge=1, le=100)
    output_latency_s: float = Field(default=0.04, gt=0, le=0.5)

    @model_validator(mode="after")
    def _mirror_differs(self) -> AudioConfig:
        mirror = self.mirror_output_device.casefold()
        if mirror and mirror == self.output_device.casefold():
            raise fail(
                "mirror_device",
                "mirror_output_device must be a different device from output_device.",
                "mirror_output_device ต้องเป็นคนละอุปกรณ์กับ output_device",
                'Use a virtual cable such as "CABLE Input" for the OBS mirror, or leave it "".',
                'ใช้สายเสียงเสมือน เช่น "CABLE Input" สำหรับ OBS หรือเว้นว่างไว้ ""',
                field="mirror_output_device",
            )
        return self


class MicConfig(_Model):
    mode: MicMode = "open"
    addressing: Literal["always", "name_or_question"] = "always"
    followup_window_s: float = Field(default=8.0, ge=0)
    read_aloud_dedupe: bool = True
    read_aloud_ratio: float = Field(default=0.75, gt=0, le=1)
    read_aloud_window_s: float = Field(default=60.0, gt=0)
    read_aloud_min_chars: int = Field(default=6, ge=1)


class VadConfig(_Model):
    backend: Literal["silero_ort", "silero_sherpa", "energy"] = "silero_ort"
    model: str = "models/vad/silero_vad.onnx"
    model_sha256: str = Field(
        default="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        pattern=r"^([0-9a-f]{64})?$",
    )
    energy_threshold_dbfs: float = Field(default=-42.0, le=0)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    neg_threshold: float = Field(default=0.35, gt=0, lt=1)
    barge_threshold: float = Field(default=0.6, gt=0, lt=1)
    min_speech_ms: int = Field(default=250, ge=0, le=5000)
    end_silence_ms: int = Field(default=600, ge=100, le=5000)
    preroll_ms: int = Field(default=300, ge=0, le=2000)
    max_segment_s: float = Field(default=15.0, gt=0)
    max_turn_s: float = Field(default=60.0, gt=0)
    particle_endpointing: bool = False
    final_particle_ms: int = Field(default=420, ge=0, le=5000)
    continuation_ms: int = Field(default=900, ge=0, le=5000)
    speculative_endpoint_ms: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _thresholds(self) -> VadConfig:
        if not self.neg_threshold < self.threshold <= self.barge_threshold:
            raise fail(
                "vad_thresholds",
                "VAD thresholds must satisfy neg_threshold < threshold <= barge_threshold.",
                "ค่า VAD ต้องเป็น neg_threshold < threshold <= barge_threshold",
                "Defaults: 0.35 < 0.5 <= 0.6 (0.65–0.7 with speakers + AEC).",
                "ค่าเริ่มต้น: 0.35 < 0.5 <= 0.6 (ใช้ 0.65–0.7 ถ้าใช้ลำโพงพร้อม AEC)",
                field="threshold",
            )
        if self.max_segment_s > self.max_turn_s:
            raise fail(
                "vad_segment",
                "max_segment_s must not exceed max_turn_s.",
                "max_segment_s ต้องไม่เกิน max_turn_s",
                field="max_segment_s",
            )
        if self.speculative_endpoint_ms and self.speculative_endpoint_ms >= self.end_silence_ms:
            raise fail(
                "vad_speculative",
                "speculative_endpoint_ms must be 0 (off) or below end_silence_ms.",
                "speculative_endpoint_ms ต้องเป็น 0 (ปิด) หรือน้อยกว่า end_silence_ms",
                field="speculative_endpoint_ms",
            )
        return self

    def endpointer_config(self) -> EndpointerConfig:
        """The voice-worker ``EndpointerConfig`` for these settings."""
        from aivtube.contracts.voice import EndpointerConfig

        return EndpointerConfig(
            threshold=self.threshold,
            neg_threshold=self.neg_threshold,
            barge_threshold=self.barge_threshold,
            min_speech_ms=self.min_speech_ms,
            end_silence_ms=self.end_silence_ms,
            preroll_ms=self.preroll_ms,
            max_segment_s=self.max_segment_s,
            max_turn_s=self.max_turn_s,
            particle_endpointing=self.particle_endpointing,
            final_particle_ms=self.final_particle_ms,
            continuation_ms=self.continuation_ms,
        )


_BACKCHANNELS = [
    "อืม",
    "อือ",
    "อ๋อ",
    "เออ",
    "ครับ",
    "ค่ะ",
    "คะ",
    "จ้ะ",
    "จ้า",
    "ฮะ",
    "โอเค",
    "เหรอ",
    "หรอ",
    "555",
    "ฮ่า",
    "ฮ่าๆ",
]


class BargeInConfig(_Model):
    policy: BargePolicy = "interrupt"
    duck_db: float = Field(default=-12.0, ge=-60, le=0)
    confirm_min_ms: int = Field(default=500, ge=0, le=5000)
    confirm_min_chars: int = Field(default=3, ge=1)
    false_timeout_s: float = Field(default=2.0, gt=0)
    cut_fade_ms: int = Field(default=60, ge=0, le=500)
    aec_warmup_s: float = Field(default=3.0, ge=0)
    boundary_ignore_s: float = Field(default=1.0, ge=0)
    quick_decode_s: float = Field(default=0.8, gt=0, le=5)
    backchannels: list[str] = Field(default_factory=lambda: list(_BACKCHANNELS))


# --- STT --------------------------------------------------------------------------------

_STT_KINDS: dict[str, frozenset[str]] = {
    "sherpa_nemo_transducer": frozenset({"model_dir", "num_threads"}),
    "pythaiasr": frozenset(),
    "openai_audio": frozenset({"base_url", "model", "api_key_env"}),
    "faster_whisper_worker": frozenset({"url", "model_dir", "compute_type"}),
    "fake": frozenset(),
}


class SttBackendConfig(_Model):
    kind: str
    enabled: bool = True
    cloud: bool = False
    model_dir: str = ""
    num_threads: int = Field(default=2, ge=1, le=16)
    base_url: str = ""
    model: str = ""
    api_key_env: str | None = None
    url: str = ""
    compute_type: str = ""

    @model_validator(mode="after")
    def _by_kind(self) -> SttBackendConfig:
        _check_kind(self.kind, _STT_KINDS, "STT backend")
        _check_kind_fields(self, self.kind, _STT_KINDS)
        _check_env_name(self.api_key_env, "api_key_env")
        if self.kind == "sherpa_nemo_transducer" and not self.model_dir:
            raise fail(
                "required", "model_dir is required.", "ต้องกำหนด model_dir", field="model_dir"
            )
        if self.kind == "openai_audio":
            _check_http_url(self.base_url, "base_url")
            if not self.model or not self.api_key_env:
                raise fail(
                    "required",
                    "openai_audio needs model and api_key_env.",
                    "openai_audio ต้องกำหนด model และ api_key_env",
                    field="model",
                )
        if self.kind == "faster_whisper_worker":
            _check_http_url(self.url, "url")
        for field in ("base_url", "url"):
            value = getattr(self, field)
            if value and not self.cloud and not _is_loopback(value):
                raise fail(
                    "cloud_flag",
                    f"{field} points off this PC; mark the backend cloud = true.",
                    f"{field} ชี้ไปนอกเครื่องนี้ ต้องตั้ง cloud = true",
                    "Cloud backends only run with privacy.cloud_stt_consent = true.",
                    "บริการคลาวด์จะทำงานเมื่อ privacy.cloud_stt_consent = true เท่านั้น",
                    field="cloud",
                )
        return self


def _default_stt_backends() -> dict[str, SttBackendConfig]:
    return {
        "typhoon_rt": SttBackendConfig(
            kind="sherpa_nemo_transducer",
            model_dir="models/stt/typhoon-asr-rt-int8",
            num_threads=2,
        ),
        "pythaiasr": SttBackendConfig(kind="pythaiasr"),
        "typhoon_api": SttBackendConfig(
            kind="openai_audio",
            cloud=True,
            base_url="https://api.opentyphoon.ai/v1",
            model="typhoon-asr-realtime",
            api_key_env="TYPHOON_API_KEY",
        ),
        "whisper_gpu": SttBackendConfig(
            kind="faster_whisper_worker",
            enabled=False,
            url="http://127.0.0.1:8091",
            model_dir="models/stt/typhoon-whisper-turbo-ct2",
            compute_type="int8_float16",
        ),
    }


class SttConfig(_Model):
    chain: list[Name] = Field(
        default_factory=lambda: ["typhoon_rt", "pythaiasr", "typhoon_api"], min_length=1
    )
    timeout_s: float = Field(default=3.0, gt=0, le=30)
    backends: dict[Name, SttBackendConfig] = Field(default_factory=_default_stt_backends)

    @model_validator(mode="after")
    def _chain(self) -> SttConfig:
        _unique(self.chain, "chain")
        for name in self.chain:
            if name not in self.backends:
                raise fail(
                    "unknown_ref",
                    f"STT chain entry {name!r} has no [stt.backends.{name}] section.",
                    f"รายการ {name!r} ใน stt.chain ไม่มีหัวข้อ [stt.backends.{name}]",
                    f"Known backends: {', '.join(sorted(self.backends))}.",
                    f"backend ที่มี: {', '.join(sorted(self.backends))}",
                    field="chain",
                )
        return self

    def effective_chain(self) -> list[str]:
        """Chain entries that are enabled (cloud entries need consent)."""
        return [n for n in self.chain if self.backends[n].enabled]


# --- LLM --------------------------------------------------------------------------------


class AutoRollbackConfig(_Model):
    window_s: float = Field(default=600.0, gt=0)
    max_failures: int = Field(default=3, ge=1)
    ttft_p95_factor: float = Field(default=2.0, gt=1)


class LlmSlotsConfig(_Model):
    speak: list[int] = Field(default_factory=lambda: [0, 1])
    background: int = Field(default=2, ge=0)
    game: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def _distinct(self) -> LlmSlotsConfig:
        slots = [*self.speak, self.background, self.game]
        if len(self.speak) != 2 or len(set(slots)) != len(slots) or min(slots) < 0:
            raise fail(
                "slots",
                "llm.slots needs two speak slots plus distinct background and game slots.",
                "llm.slots ต้องมี speak 2 ช่อง และ background กับ game ที่ไม่ซ้ำกัน",
                "Default: speak = [0, 1], background = 2, game = 3.",
                "ค่าเริ่มต้น: speak = [0, 1], background = 2, game = 3",
            )
        return self

    def highest(self, *, games: bool) -> int:
        return max([*self.speak, self.background, *([self.game] if games else [])])


_MANAGED_LLAMA_FLAGS = frozenset(
    {
        "--port",
        "--host",
        "-m",
        "--model",
        "-a",
        "--alias",
        "--slot-save-path",
        "-c",
        "--ctx-size",
        "-np",
        "--parallel",
        "-t",
        "--threads",
        "--fit",
        "-fit",
        "--fit-target",
        "--n-cpu-moe",
        "-ncmoe",
        "--cpu-moe",
        "-cmoe",
    }
)


class LlamaServerConfig(_Model):
    exe: str = "vendor/llama.cpp/llama-server.exe"
    model: str
    alias: str = Field(min_length=1)
    port: Port
    autostart: Literal["always", "on_demand", "never"] = "on_demand"
    adopt_existing: bool = True
    ctx: int = Field(default=16384, ge=512, le=262144)
    parallel: int = Field(default=3, ge=1, le=16)
    placement: Literal["fit", "pinned", "all_gpu"] = "fit"
    fit_target_mib: int = Field(default=3584, ge=0)
    pinned_n_cpu_moe: int = Field(default=0, ge=0)
    n_cpu_moe_step: int = Field(default=2, ge=1)
    threads: int = Field(default=8, ge=1, le=64)
    slot_save_dir: str = "data/kv"
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("extra_args")
    @classmethod
    def _unmanaged(cls, value: list[str]) -> list[str]:
        for arg in value:
            flag = arg.split("=", 1)[0]
            if flag in _MANAGED_LLAMA_FLAGS:
                raise fail(
                    "managed_flag",
                    f"{flag} is set by aivtube from its own keys; remove it from extra_args.",
                    f"{flag} ถูกกำหนดโดย aivtube จากคีย์ของมันเองแล้ว ให้ลบออกจาก extra_args",
                    "Use port, model, alias, ctx, parallel, threads or placement instead.",
                    "ใช้คีย์ port, model, alias, ctx, parallel, threads หรือ placement แทน",
                )
        return value


_LLM_KINDS: dict[str, frozenset[str]] = {
    "llamacpp": frozenset(
        {
            "server",
            "flavor",
            "base_url",
            "model",
            "api_key_env",
            "first_token_timeout_s",
            "extra_body",
        }
    ),
    "openai_compat": frozenset(
        {
            "flavor",
            "base_url",
            "model",
            "api_key_env",
            "first_token_timeout_s",
            "extra_body",
            "reasoning_effort",
        }
    ),
    "canned": frozenset(),
    "fake": frozenset({"first_token_timeout_s"}),
}


class LlmProviderConfig(_Model):
    kind: str
    enabled: bool = True
    cloud: bool = False
    server: str | None = None
    flavor: Literal["llamacpp", "typhoon", "gemini", "openai"] | None = None
    base_url: str = ""
    model: str = ""
    api_key_env: str | None = None
    first_token_timeout_s: float = Field(default=4.0, gt=0, le=120)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _by_kind(self) -> LlmProviderConfig:
        _check_kind(self.kind, _LLM_KINDS, "LLM provider")
        _check_kind_fields(self, self.kind, _LLM_KINDS)
        _check_env_name(self.api_key_env, "api_key_env")
        if self.kind in ("llamacpp", "openai_compat"):
            _check_http_url(self.base_url, "base_url")
            if not self.model:
                raise fail("required", "model is required.", "ต้องกำหนด model", field="model")
        if self.kind == "llamacpp":
            if not self.server:
                raise fail(
                    "required",
                    'A llamacpp provider needs server = "<name in llm.servers>".',
                    'ผู้ให้บริการชนิด llamacpp ต้องกำหนด server = "<ชื่อใน llm.servers>"',
                    field="server",
                )
            if self.cloud:
                raise fail(
                    "cloud_flag",
                    "A llamacpp provider runs on this PC; cloud must be false.",
                    "llamacpp ทำงานบนเครื่องนี้ ต้องตั้ง cloud = false",
                    field="cloud",
                )
        if self.base_url and not self.cloud and not _is_loopback(self.base_url):
            raise fail(
                "cloud_flag",
                "base_url points off this PC; mark the provider cloud = true.",
                "base_url ชี้ไปนอกเครื่องนี้ ต้องตั้ง cloud = true",
                "Cloud providers only run with privacy.cloud_llm_consent = true.",
                "ผู้ให้บริการคลาวด์จะทำงานเมื่อ privacy.cloud_llm_consent = true เท่านั้น",
                field="cloud",
            )
        return self

    @property
    def effective_flavor(self) -> str:
        if self.flavor:
            return self.flavor
        return "llamacpp" if self.kind == "llamacpp" else "openai"


_SAMPLING_ARGS = [
    "--temp",
    "0.6",
    "--top-p",
    "0.95",
    "--repeat-penalty",
    "1.05",
    "--cache-reuse",
    "256",
]


def _default_llm_servers() -> dict[str, LlamaServerConfig]:
    return {
        "local30b": LlamaServerConfig(
            model="models/llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf",
            alias="pailin-30b",
            port=8080,
            autostart="always",
            ctx=24576,
            parallel=3,
            placement="fit",
            fit_target_mib=3584,
            extra_args=[
                "-kvu",
                "-fa",
                "on",
                "-ctk",
                "q8_0",
                "-ctv",
                "q8_0",
                "-lm",
                "none",
                *_SAMPLING_ARGS,
                "--metrics",
            ],
        ),
        "local4b": LlamaServerConfig(
            model="models/llm/typhoon2.5-qwen3-4b.Q4_K_M.gguf",
            alias="pailin-4b",
            port=8081,
            autostart="on_demand",
            ctx=16384,
            parallel=3,
            placement="all_gpu",
            extra_args=[
                "-kvu",
                "-ngl",
                "all",
                "-fa",
                "on",
                "-ctk",
                "q8_0",
                "-ctv",
                "q8_0",
                *_SAMPLING_ARGS,
            ],
        ),
    }


def _default_llm_providers() -> dict[str, LlmProviderConfig]:
    return {
        "local-30b": LlmProviderConfig(
            kind="llamacpp",
            server="local30b",
            base_url="http://127.0.0.1:8080/v1",
            model="pailin-30b",
            first_token_timeout_s=4.0,
        ),
        "local-4b": LlmProviderConfig(
            kind="llamacpp",
            server="local4b",
            base_url="http://127.0.0.1:8081/v1",
            model="pailin-4b",
            first_token_timeout_s=2.0,
        ),
        "typhoon-api": LlmProviderConfig(
            kind="openai_compat",
            flavor="typhoon",
            cloud=True,
            base_url="https://api.opentyphoon.ai/v1",
            model="typhoon-v2.5-30b-a3b-instruct",
            api_key_env="TYPHOON_API_KEY",
            first_token_timeout_s=6.0,
            extra_body={"repetition_penalty": 1.05},
        ),
        "gemini": LlmProviderConfig(
            kind="openai_compat",
            flavor="gemini",
            cloud=True,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-3.5-flash-lite",
            api_key_env="GEMINI_API_KEY",
            first_token_timeout_s=8.0,
            reasoning_effort="minimal",
        ),
    }


class LlmConfig(_Model):
    chain: list[Name] = Field(
        default_factory=lambda: ["local-30b", "local-4b", "typhoon-api", "gemini"], min_length=1
    )
    canned_line: str = Field(default="เอ๊ะ สมองไพลินค้างแป๊บนึงนะ", min_length=1)
    max_reply_tokens: int = Field(default=256, ge=16, le=4096)
    temperature: float = Field(default=0.6, ge=0, le=2)
    strict_temperature: float = Field(default=0.4, ge=0, le=2)
    hedge_after_ms: int = Field(default=0, ge=0)
    auto_return_after_s: float = Field(default=60.0, ge=0)
    connect_timeout_s: float = Field(default=2.0, gt=0, le=30)
    stall_timeout_s: float = Field(default=3.0, gt=0, le=60)
    breaker_max_s: float = Field(default=60.0, gt=0)
    auto_rollback: AutoRollbackConfig = Field(default_factory=AutoRollbackConfig)
    slots: LlmSlotsConfig = Field(default_factory=LlmSlotsConfig)
    servers: dict[Name, LlamaServerConfig] = Field(default_factory=_default_llm_servers)
    providers: dict[Name, LlmProviderConfig] = Field(default_factory=_default_llm_providers)

    @model_validator(mode="after")
    def _refs(self) -> LlmConfig:
        _unique(self.chain, "chain")
        for name in self.chain:
            if name not in self.providers:
                raise fail(
                    "unknown_ref",
                    f"LLM chain entry {name!r} has no [llm.providers.{name}] section.",
                    f"รายการ {name!r} ใน llm.chain ไม่มีหัวข้อ [llm.providers.{name}]",
                    f"Known providers: {', '.join(sorted(self.providers))}.",
                    f"ผู้ให้บริการที่มี: {', '.join(sorted(self.providers))}",
                    field="chain",
                )
        for name, prov in self.providers.items():
            if prov.server is None:
                continue
            server = self.servers.get(prov.server)
            if server is None:
                raise fail(
                    "unknown_ref",
                    f"Provider {name!r} uses server {prov.server!r}, which is not in llm.servers.",
                    f"ผู้ให้บริการ {name!r} อ้างถึงเซิร์ฟเวอร์ {prov.server!r} ที่ไม่มีใน llm.servers",
                    path=f"llm.providers.{name}.server",
                )
            _, host, port = _url_port(prov.base_url)
            if host in LOOPBACK_HOSTS and port != server.port:
                raise fail(
                    "port_mismatch",
                    f"Provider {name!r} base_url uses port {port}, but server {prov.server!r} "
                    f"listens on {server.port}.",
                    f"base_url ของ {name!r} ใช้พอร์ต {port} แต่เซิร์ฟเวอร์ {prov.server!r} ใช้พอร์ต "
                    f"{server.port}",
                    "Make the two ports match.",
                    "ตั้งพอร์ตทั้งสองให้ตรงกัน",
                    path=f"llm.providers.{name}.base_url",
                    also=f"llm.servers.{prov.server}.port",
                )
        return self

    def effective_chain(self) -> list[str]:
        """Chain entries that are enabled (cloud entries need consent)."""
        return [n for n in self.chain if self.providers[n].enabled]


# --- TTS --------------------------------------------------------------------------------

_RATE = re.compile(r"^[+-]\d{1,3}%$")
_PITCH = re.compile(r"^[+-]\d{1,4}Hz$")


class TtsIdentityConfig(_Model):
    voice: str = Field(min_length=1)
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    backends: list[Name] = Field(min_length=1)

    @model_validator(mode="after")
    def _prosody(self) -> TtsIdentityConfig:
        for field, pattern, example in (
            ("rate", _RATE, "+8%"),
            ("volume", _RATE, "+0%"),
            ("pitch", _PITCH, "+20Hz"),
        ):
            if not pattern.match(getattr(self, field)):
                raise fail(
                    "prosody",
                    f"{field} must look like {example!r}.",
                    f"{field} ต้องอยู่ในรูปแบบ {example!r}",
                    field=field,
                )
        _unique(self.backends, "backends")
        return self

    def voice_spec(self, identity: str) -> VoiceSpec:
        return VoiceSpec(
            identity=identity,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume,
        )


_TTS_KINDS: dict[str, frozenset[str]] = {
    "edge": frozenset(),
    "azure": frozenset(
        {
            "key_env",
            "region_env",
            "tier",
            "requests_per_min",
            "min_chars_when_primary_f0",
            "hedge_first_chunk",
        }
    ),
    "piper": frozenset({"lexicon"}),
    "captions": frozenset(),
    "fake": frozenset(),
}


class TtsBackendConfig(_Model):
    kind: str
    enabled: bool = True
    key_env: str | None = None
    region_env: str | None = None
    tier: Literal["F0", "S0"] = "F0"
    requests_per_min: int = Field(default=18, ge=1, le=10000)
    min_chars_when_primary_f0: int = Field(default=60, ge=1, le=400)
    hedge_first_chunk: bool = False
    lexicon: str = ""

    @model_validator(mode="after")
    def _by_kind(self) -> TtsBackendConfig:
        _check_kind(self.kind, _TTS_KINDS, "TTS backend")
        _check_kind_fields(self, self.kind, _TTS_KINDS)
        _check_env_name(self.key_env, "key_env")
        _check_env_name(self.region_env, "region_env")
        if self.kind == "azure":
            if not self.key_env or not self.region_env:
                raise fail(
                    "required",
                    "Azure needs key_env and region_env.",
                    "Azure ต้องกำหนด key_env และ region_env",
                    field="key_env",
                )
            if self.tier == "F0" and self.hedge_first_chunk:
                raise fail(
                    "azure_f0_hedge",
                    "hedge_first_chunk needs Azure tier S0; F0 allows only 20 requests per minute.",
                    "hedge_first_chunk ใช้ได้กับ Azure แบบ S0 เท่านั้น เพราะ F0 จำกัดแค่ 20 ครั้งต่อนาที",
                    'Set hedge_first_chunk = false, or tier = "S0" if you pay for Azure.',
                    'ตั้ง hedge_first_chunk = false หรือ tier = "S0" ถ้าใช้ Azure แบบเสียเงิน',
                    field="hedge_first_chunk",
                )
            if self.tier == "F0" and self.requests_per_min > 20:
                raise fail(
                    "azure_f0_quota",
                    "Azure F0 allows at most 20 requests per minute.",
                    "Azure F0 รับได้ไม่เกิน 20 ครั้งต่อนาที",
                    "Keep requests_per_min at 18.",
                    "ตั้ง requests_per_min = 18",
                    field="requests_per_min",
                )
        return self


def _default_tts_identities() -> dict[str, TtsIdentityConfig]:
    return {
        "premwadee": TtsIdentityConfig(
            voice="th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz", backends=["edge", "azure"]
        ),
        "offline": TtsIdentityConfig(
            voice="models/tts/th_TH-tsync2-medium.onnx", backends=["piper"]
        ),
    }


def _default_tts_backends() -> dict[str, TtsBackendConfig]:
    return {
        "edge": TtsBackendConfig(kind="edge"),
        "azure": TtsBackendConfig(
            kind="azure", key_env="AZURE_SPEECH_KEY", region_env="AZURE_SPEECH_REGION"
        ),
        "piper": TtsBackendConfig(kind="piper", lexicon="characters/{character}/lexicon.toml"),
    }


class ChunkerSection(_Model):
    first_min_chars: int = Field(default=8, ge=1)
    first_max_chars: int = Field(default=60, ge=1)
    min_chars: int = Field(default=40, ge=1)
    max_chars: int = Field(default=160, ge=1, le=400)
    strong_min_chars: int = Field(default=20, ge=1)
    stall_flush_ms: int = Field(default=500, ge=50, le=5000)

    @model_validator(mode="after")
    def _ranges(self) -> ChunkerSection:
        if not (
            self.first_min_chars <= self.first_max_chars <= self.max_chars
            and self.min_chars <= self.max_chars
        ):
            raise fail(
                "chunker",
                "Need first_min_chars <= first_max_chars <= max_chars and min_chars <= max_chars.",
                "ต้องเป็น first_min_chars <= first_max_chars <= max_chars และ min_chars <= max_chars",
                "Defaults: 8 / 60 / 40 / 160.",
                "ค่าเริ่มต้น: 8 / 60 / 40 / 160",
            )
        return self


class TtsConfig(_Model):
    identity_chain: list[Name] = Field(default_factory=lambda: ["premwadee"], min_length=1)
    captions_fallback: bool = True
    first_audio_timeout_s: float = Field(default=2.0, gt=0, le=30)
    later_timeout_s: float = Field(default=4.0, gt=0, le=30)
    first_retries: int = Field(default=0, ge=0, le=3)
    later_retries: int = Field(default=1, ge=0, le=3)
    max_in_flight: int = Field(default=2, ge=1, le=4)
    max_queued_segments: int = Field(default=8, ge=1, le=8)
    filler_after_s: float = Field(default=1.2, gt=0)
    filler_min_interval_s: float = Field(default=30.0, ge=0)
    breaker_failures: int = Field(default=3, ge=1)
    breaker_window_s: float = Field(default=60.0, gt=0)
    breaker_cooldown_s: float = Field(default=120.0, gt=0)
    cache_dir: str = "data/cache/tts"
    identities: dict[Name, TtsIdentityConfig] = Field(default_factory=_default_tts_identities)
    backends: dict[Name, TtsBackendConfig] = Field(default_factory=_default_tts_backends)
    chunker: ChunkerSection = Field(default_factory=ChunkerSection)

    @model_validator(mode="after")
    def _refs(self) -> TtsConfig:
        _unique(self.identity_chain, "identity_chain")
        for name in self.identity_chain:
            if name not in self.identities:
                raise fail(
                    "unknown_ref",
                    f"TTS identity {name!r} has no [tts.identities.{name}] section.",
                    f"เสียง {name!r} ไม่มีหัวข้อ [tts.identities.{name}]",
                    f"Known identities: {', '.join(sorted(self.identities))}.",
                    f"เสียงที่มี: {', '.join(sorted(self.identities))}",
                    field="identity_chain",
                )
        for ident, cfg in self.identities.items():
            for backend in cfg.backends:
                if backend not in self.backends:
                    raise fail(
                        "unknown_ref",
                        f"Identity {ident!r} uses backend {backend!r}, which is not in "
                        "tts.backends.",
                        f"เสียง {ident!r} อ้างถึง backend {backend!r} ที่ไม่มีใน tts.backends",
                        path=f"tts.identities.{ident}.backends",
                    )
        return self


# --- brain, chat, memory, safety, avatar, panel, tools, games ---------------------------


class BrainBudget(_Model):
    static_tokens: int = Field(default=1800, ge=100)
    context_tokens: int = Field(default=1500, ge=0)
    history_tokens: int = Field(default=5000, ge=0)
    tail_voice_tokens: int = Field(default=250, ge=50)
    tail_chat_tokens: int = Field(default=450, ge=50)
    cache_alarm_ratio: float = Field(default=0.85, ge=0, le=1)


class BrainConfig(_Model):
    post_speech_gap_s: float = Field(default=0.3, ge=0)
    chat_min_interval_s: float = Field(default=4.0, ge=0)
    chat_min_interval_voice_s: float = Field(default=8.0, ge=0)
    chat_gather_s: float = Field(default=1.0, ge=0)
    chat_k: int = Field(default=3, ge=1, le=10)
    idle_after_s: float = Field(default=25.0, gt=0)
    idle_jitter_s: float = Field(default=5.0, ge=0)
    idle_max_s: float = Field(default=120.0, gt=0)
    decision_watchdog_s: float = Field(default=30.0, gt=0)
    max_tool_rounds: int = Field(default=2, ge=1, le=4)
    user_speaking_watchdog_s: float = Field(default=20.0, gt=0)
    opener_repeat_ratio: float = Field(default=0.3, gt=0, le=1)
    critical_cut_wait_ms: int = Field(default=150, ge=0, le=2000)
    budget: BrainBudget = Field(default_factory=BrainBudget)

    @model_validator(mode="after")
    def _idle(self) -> BrainConfig:
        if not self.idle_jitter_s < self.idle_after_s <= self.idle_max_s:
            raise fail(
                "idle",
                "Need idle_jitter_s < idle_after_s <= idle_max_s.",
                "ต้องเป็น idle_jitter_s < idle_after_s <= idle_max_s",
                field="idle_after_s",
            )
        return self


_CHAT_SOURCES = frozenset(
    {
        "twitch_irc",
        "youtube_poll",
        "fake",
        "twitch_eventsub",
        "youtube_grpc",
        "alerts",
        "tiktok",
    }
)


class TwitchIrcConfig(_Model):
    channel: str = ""
    url: str = "wss://irc-ws.chat.twitch.tv:443"

    @field_validator("channel", mode="before")
    @classmethod
    def _channel(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        name = value.strip().lstrip("#").lower()
        if name and not re.fullmatch(r"[a-z0-9_]{3,25}", name):
            raise fail(
                "twitch_channel",
                f"{value!r} is not a Twitch channel name.",
                f"{value!r} ไม่ใช่ชื่อช่อง Twitch",
                "Use the login name from twitch.tv/<name>.",
                "ใช้ชื่อจากลิงก์ twitch.tv/<ชื่อ>",
            )
        return name


class YouTubePollConfig(_Model):
    api_key_env: str = "YOUTUBE_API_KEY"
    video_id: str = ""
    handle: str = ""
    min_interval_s: float = Field(default=2.0, ge=1.0)


class ChatWindowConfig(_Model):
    horizon_s: float = Field(default=40.0, gt=0)
    must_ack_ttl_s: float = Field(default=600.0, gt=0)
    temperature: float = Field(default=0.6, gt=0)
    user_cooldown_s: float = Field(default=60.0, ge=0)
    dedupe_lru: int = Field(default=5000, ge=100)


class ChatConfig(_Model):
    enabled: bool = True
    sources: list[str] = Field(default_factory=lambda: ["twitch_irc"])
    max_msg_chars: int = Field(default=300, ge=20, le=2000)
    silence_timeout_s: float = Field(default=300.0, gt=0)
    restart_backoff_s: Backoff = Field(default_factory=lambda: [1.0, 30.0])
    twitch_irc: TwitchIrcConfig = Field(default_factory=TwitchIrcConfig)
    youtube_poll: YouTubePollConfig = Field(default_factory=YouTubePollConfig)
    window: ChatWindowConfig = Field(default_factory=ChatWindowConfig)

    @field_validator("sources")
    @classmethod
    def _sources(cls, value: list[str]) -> list[str]:
        _unique(value, "sources")
        for src in value:
            if src not in _CHAT_SOURCES and not _PLUGIN_KIND.match(src):
                raise fail(
                    "chat_source",
                    f"Unknown chat source {src!r}.",
                    f"ไม่รู้จักแหล่งแชท {src!r}",
                    f"Built-in sources: {', '.join(sorted(_CHAT_SOURCES))}.",
                    f"แหล่งแชทที่มี: {', '.join(sorted(_CHAT_SOURCES))}",
                )
        return value


class MemoryConfig(_Model):
    core_slots: int = Field(default=16, ge=1, le=64)
    slot_max_chars: int = Field(default=120, ge=20, le=500)
    max_writes_per_session: int = Field(default=8, ge=0)
    min_write_interval_s: float = Field(default=120.0, ge=0)
    chat_sourced: Literal["quarantine", "allow"] = "quarantine"
    recall_k: int = Field(default=2, ge=0, le=10)
    viewer_facts_per_viewer: int = Field(default=3, ge=0)
    viewer_facts_max_viewers: int = Field(default=5, ge=0)
    episode_after_idle_s: float = Field(default=1800.0, gt=0)
    backup_keep: int = Field(default=14, ge=1)
    backup_dir: str = "data/backups"
    ops_db: str = "data/ops.db"


class SafetyConfig(_Model):
    base_lists: str = "config/filters/base"
    private_lists: str = "config/filters/private"
    platform_overlays: dict[Platform, str] = Field(default_factory=dict)
    prev_tail_chars: int = Field(default=40, ge=0, le=400)
    politics: Literal["review", "block", "allow"] = "review"
    mask_text: str = "[ลิงก์]"
    tier1: Literal["off", "on", "strict"] = "off"
    tier1_block: float = Field(default=0.8, ge=0, le=1)
    tier1_review: float = Field(default=0.5, ge=0, le=1)
    tier1_timeout_ms: int = Field(default=300, ge=10)
    output_hold_ms: int = Field(default=250, ge=0)
    fail_closed_categories: list[str] = Field(default_factory=lambda: ["monarchy_112"])
    auto_strict_after: int = Field(default=3, ge=0)
    auto_strict_window_s: float = Field(default=300.0, gt=0)
    auto_strict_mute_s: float = Field(default=600.0, ge=0)
    auto_freeze_after: int = Field(default=0, ge=0)
    filtered_text: str = Field(default="Filtered.", min_length=1)
    review_mode: bool = False

    @model_validator(mode="after")
    def _rules(self) -> SafetyConfig:
        if self.tier1_review > self.tier1_block:
            raise fail(
                "tier1",
                "tier1_review must not exceed tier1_block.",
                "tier1_review ต้องไม่เกิน tier1_block",
                field="tier1_review",
            )
        if "monarchy_112" not in self.fail_closed_categories:
            raise fail(
                "fail_closed",
                "monarchy_112 must stay fail-closed (Thai lèse-majesté law).",
                "หมวด monarchy_112 ต้องเป็นแบบ fail-closed เสมอ (กฎหมายมาตรา 112)",
                'Keep "monarchy_112" in fail_closed_categories.',
                'คง "monarchy_112" ไว้ใน fail_closed_categories',
                field="fail_closed_categories",
            )
        return self


class AvatarConfig(_Model):
    sink: Literal["vts", "browser", "none"] = "vts"
    fps: int = Field(default=60, ge=1, le=120)
    lead_ms: int = Field(default=40, ge=0, le=500)
    jitter_fallback_fps: int = Field(default=30, ge=1, le=120)
    jitter_limit_ms: float = Field(default=8.0, gt=0)
    idle_motion: bool = True
    emotion_fade_s: float = Field(default=0.3, ge=0)
    request_timeout_s: float = Field(default=2.0, gt=0)
    max_inflight: int = Field(default=8, ge=1, le=64)
    reconnect_backoff_s: Backoff = Field(default_factory=lambda: [1.0, 10.0])
    discovery_timeout_s: float = Field(default=2.5, gt=0)

    @model_validator(mode="after")
    def _fps(self) -> AvatarConfig:
        if self.jitter_fallback_fps > self.fps:
            raise fail(
                "fps",
                "jitter_fallback_fps must not exceed fps.",
                "jitter_fallback_fps ต้องไม่เกิน fps",
                field="jitter_fallback_fps",
            )
        return self


class PanelConfig(_Model):
    host: str = "127.0.0.1"
    open_browser: bool = True
    ws_max_hz: float = Field(default=20.0, gt=0, le=120)

    @field_validator("host")
    @classmethod
    def _loopback(cls, value: str) -> str:
        if value not in LOOPBACK_HOSTS:
            raise fail(
                "loopback",
                f"The panel must bind a loopback address, not {value!r}.",
                f"แผงควบคุมต้องผูกกับที่อยู่ภายในเครื่องเท่านั้น ไม่ใช่ {value!r}",
                "Use 127.0.0.1 (OBS docks and the browser run on this PC).",
                "ใช้ 127.0.0.1 (OBS และเบราว์เซอร์อยู่บนเครื่องเดียวกัน)",
            )
        return value


class ToolsConfig(_Model):
    mode: Literal["live", "dry_run", "off"] = "live"
    approval_timeout_s: float = Field(default=20.0, gt=0)
    default_timeout_s: float = Field(default=5.0, gt=0)
    timeout_max_s: int = Field(default=600, ge=1, le=600)


class GamesConfig(_Model):
    enabled: bool = False
    force_max_retries: int = Field(default=3, ge=0, le=10)
    result_timeout_s: float = Field(default=20.0, gt=0)
    action_strategy: Literal["auto", "grammar", "tools"] = "auto"


class LoggingConfig(_Model):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir: str = "logs"
    max_mb: int = Field(default=20, ge=1, le=1024)
    backups: int = Field(default=5, ge=0, le=100)
    keep_days: int = Field(default=14, ge=1)
    jsonl: bool = True
    console: bool = True

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


class LauncherConfig(_Model):
    keep_llm: bool = True
    core_timeout_s: float = Field(default=5.0, gt=0)
    graceful_timeout_s: float = Field(default=5.0, gt=0)
    restart_backoff_s: Backoff = Field(default_factory=lambda: [0.5, 30.0])
    crash_loop_restarts: int = Field(default=5, ge=1)
    crash_loop_window_s: float = Field(default=120.0, gt=0)
    llm_health_interval_s: float = Field(default=2.0, gt=0)
    llm_health_failures: int = Field(default=3, ge=1)
    llm_load_timeout_s: float = Field(default=300.0, gt=0)
    llm_max_load_failures: int = Field(default=3, ge=1)
    vram_alarm_mib: int = Field(default=500, ge=0)
    gpu_poll_s: float = Field(default=5.0, gt=0)
    keep_awake: bool = True


# --- AppConfig --------------------------------------------------------------------------


class AppConfig(_Model):
    """The whole validated config. Load it with ``aivtube.config.load_config``."""

    schema_version: int = CURRENT_SCHEMA_VERSION
    active_profile: str = "stream"
    characters: list[CharId] = Field(default_factory=lambda: ["pailin"], min_length=1)
    app: AppSection = Field(default_factory=AppSection)
    ports: PortsConfig = Field(default_factory=PortsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    mic: MicConfig = Field(default_factory=MicConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)
    panel: PanelConfig = Field(default_factory=PanelConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    games: GamesConfig = Field(default_factory=GamesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    launcher: LauncherConfig = Field(default_factory=LauncherConfig)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    character_overrides: dict[CharId, dict[str, Any]] = Field(default_factory=dict)

    _root: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _cross_checks(self) -> AppConfig:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise fail(
                "schema_version",
                f"schema_version {self.schema_version} is not supported (expected "
                f"{CURRENT_SCHEMA_VERSION}).",
                f"ไม่รองรับ schema_version {self.schema_version} (ต้องเป็น {CURRENT_SCHEMA_VERSION})",
                "Run: aivtube config migrate",
                "รันคำสั่ง: aivtube config migrate",
                field="schema_version",
            )
        _unique(self.characters, "characters")
        if self.active_profile not in self.profiles and self.active_profile != "stream":
            known = ", ".join(sorted({*self.profiles, "stream"}))
            raise fail(
                "profile",
                f"Unknown profile {self.active_profile!r}. Known profiles: {known}.",
                f"ไม่รู้จักโปรไฟล์ {self.active_profile!r} โปรไฟล์ที่มี: {known}",
                field="active_profile",
            )
        self._check_ports()
        self._check_slots()
        self._apply_consent()
        return self

    def port_items(self) -> list[tuple[str, int]]:
        """Every TCP port this config binds or connects to locally, as (key, port)."""
        items = [(f"ports.{name}", int(port)) for name, port in self.ports]
        items += [(f"llm.servers.{name}.port", s.port) for name, s in self.llm.servers.items()]
        for name, backend in self.stt.backends.items():
            if backend.url and _is_loopback(backend.url):
                port = _url_port(backend.url)[2]
                if port is not None:
                    items.append((f"stt.backends.{name}.url", port))
        return items

    def _check_ports(self) -> None:
        seen: dict[int, str] = {}
        for path, port in self.port_items():
            if port in VTS_PORT_RANGE:
                raise fail(
                    "port_vts_reserved",
                    f"Port {port} is reserved for VTube Studio (8001–8009).",
                    f"พอร์ต {port} สงวนไว้ให้ VTube Studio (8001–8009)",
                    "Use another port.",
                    "ใช้พอร์ตอื่น",
                    path=path,
                )
            if port in seen:
                raise fail(
                    "port_duplicate",
                    f"Port {port} is used twice: {seen[port]} and {path}.",
                    f"พอร์ต {port} ถูกใช้ซ้ำ: {seen[port]} และ {path}",
                    "Give every service its own port (see ARCHITECTURE §2.2).",
                    "กำหนดพอร์ตให้แต่ละบริการไม่ซ้ำกัน",
                    path=path,
                    also=seen[port],
                )
            seen[port] = path

    def _check_slots(self) -> None:
        needed = self.llm.slots.highest(games=self.games.enabled) + 1
        used = {p.server for p in self.llm.providers.values() if p.server}
        for name in sorted(used):
            server = self.llm.servers[name]
            if server.parallel < needed:
                raise fail(
                    "llm_parallel",
                    f"Server {name!r} has parallel = {server.parallel}, but llm.slots needs "
                    f"{needed} slots.",
                    f"เซิร์ฟเวอร์ {name!r} ตั้ง parallel = {server.parallel} แต่ llm.slots ต้องใช้ {needed} ช่อง",
                    f"Set parallel = {needed} (games add slot {self.llm.slots.game}).",
                    f"ตั้ง parallel = {needed}",
                    path=f"llm.servers.{name}.parallel",
                )

    def _apply_consent(self) -> None:
        if not self.privacy.cloud_llm_consent:
            off = {
                n: p.model_copy(update={"enabled": False})
                for n, p in self.llm.providers.items()
                if p.cloud and p.enabled
            }
            if off:
                llm = self.llm.model_copy(update={"providers": {**self.llm.providers, **off}})
                object.__setattr__(self, "llm", llm)
        if not self.privacy.cloud_stt_consent:
            off_stt = {
                n: b.model_copy(update={"enabled": False})
                for n, b in self.stt.backends.items()
                if b.cloud and b.enabled
            }
            if off_stt:
                stt = self.stt.model_copy(update={"backends": {**self.stt.backends, **off_stt}})
                object.__setattr__(self, "stt", stt)

    # -- helpers -------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The project root this config was loaded from (the CWD if built in code)."""
        return self._root if self._root is not None else Path.cwd()

    def resolve_path(self, value: str | Path, *, character: str | None = None) -> Path:
        """Expand ``{character}`` and make ``value`` absolute against :attr:`root`."""
        text = str(value)
        if CHARACTER_PLACEHOLDER in text:
            if character is None:
                raise ValueError(f"{text!r} needs a character id")
            text = expand_character(text, character)
        path = Path(text)
        return path if path.is_absolute() else self.root / path

    def consent_blocked(self) -> list[str]:
        """Cloud providers/backends switched off because consent is missing."""
        out: list[str] = []
        if not self.privacy.cloud_llm_consent:
            out += [f"llm.providers.{n}" for n, p in self.llm.providers.items() if p.cloud]
        if not self.privacy.cloud_stt_consent:
            out += [f"stt.backends.{n}" for n, b in self.stt.backends.items() if b.cloud]
        return out

    def tts_chain_for(self, character: CharacterConfig) -> list[str]:
        """The character's identities first, then app-level ones it does not already list."""
        chain = list(character.tts_identity_chain)
        chain += [i for i in self.tts.identity_chain if i not in chain]
        return chain

    def profile_names(self) -> list[str]:
        return sorted({*self.profiles, "stream"})


# --- characters -------------------------------------------------------------------------


class EmotionPose(_Model):
    expressions: list[str] = Field(default_factory=list)
    smile: float | None = Field(default=None, ge=0, le=1)
    brows: float | None = Field(default=None, ge=0, le=1)
    params: dict[str, float] = Field(default_factory=dict)


class CharacterAvatarConfig(_Model):
    vts_url: str = "ws://127.0.0.1:8001"
    vts_window_title: str = ""
    plugin_name: str = Field(default="AI_Vtube Brain", min_length=3, max_length=32)
    plugin_developer: str = Field(default="AI_Vtube", min_length=3, max_length=32)
    token_file: str = "data/tokens/vts_{character}.txt"
    emotion_map: dict[str, EmotionPose] = Field(default_factory=dict)

    @field_validator("vts_url")
    @classmethod
    def _ws(cls, value: str) -> str:
        scheme, host, _ = _url_port(value)
        if scheme not in ("ws", "wss") or not host:
            raise fail(
                "url",
                f"vts_url must be a ws:// URL, got {value!r}.",
                f"vts_url ต้องเป็น URL แบบ ws:// แต่ได้ {value!r}",
                "Default: ws://127.0.0.1:8001",
                "ค่าเริ่มต้น: ws://127.0.0.1:8001",
            )
        return value


class CharacterMemoryConfig(_Model):
    db: str = "data/memory/{character}.sqlite"


class CharacterSafetyConfig(_Model):
    overlay: str = "filters.toml"


class CharacterGamesConfig(_Model):
    port: Port = 8000
    character_id: str = "{character}"


class CharacterToolsConfig(_Model):
    enabled: list[str] = Field(default_factory=lambda: ["remember", "forget"])

    @field_validator("enabled")
    @classmethod
    def _tools(cls, value: list[str]) -> list[str]:
        _unique(value, "enabled")
        return value


class CharacterConfig(_Model):
    """One character (``characters/<id>/character.toml``). Load with ``load_character``."""

    id: CharId
    display_name: str = Field(min_length=1, max_length=32)
    name_th: str = ""
    persona: str = "persona.th.md"
    lexicon: str = "lexicon.toml"
    aliases: list[str] = Field(min_length=1)
    tts_identity_chain: list[Name] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=lambda: list(EMOTIONS))
    cached_phrases: list[str] = Field(default_factory=list)
    stt_aliases: dict[str, list[str]] = Field(default_factory=dict)
    avatar: CharacterAvatarConfig = Field(default_factory=CharacterAvatarConfig)
    memory: CharacterMemoryConfig = Field(default_factory=CharacterMemoryConfig)
    safety: CharacterSafetyConfig = Field(default_factory=CharacterSafetyConfig)
    games: CharacterGamesConfig = Field(default_factory=CharacterGamesConfig)
    tools: CharacterToolsConfig = Field(default_factory=CharacterToolsConfig)

    _root: Path | None = PrivateAttr(default=None)
    _dir: Path | None = PrivateAttr(default=None)

    @field_validator("display_name")
    @classmethod
    def _ascii(cls, value: str) -> str:
        if not value.isascii() or not value.isprintable():
            raise fail(
                "display_name",
                "display_name must be printable ASCII (game SDKs and VTS show it).",
                "display_name ต้องเป็นอักษรภาษาอังกฤษ (ASCII) เพราะเกมและ VTS ใช้แสดงผล",
                "Put the Thai name in name_th.",
                "ใส่ชื่อภาษาไทยไว้ใน name_th",
            )
        return value

    @model_validator(mode="after")
    def _emotions(self) -> CharacterConfig:
        _unique(self.emotions, "emotions")
        _unique(self.tts_identity_chain, "tts_identity_chain")
        if "neutral" not in self.emotions:
            raise fail(
                "emotions",
                'emotions must include "neutral".',
                'emotions ต้องมี "neutral"',
                field="emotions",
            )
        for name in self.avatar.emotion_map:
            if name not in self.emotions:
                raise fail(
                    "emotion_map",
                    f"emotion_map has {name!r}, which is not in emotions.",
                    f"emotion_map มี {name!r} ที่ไม่อยู่ใน emotions",
                    path=f"avatar.emotion_map.{name}",
                )
        if CHARACTER_PLACEHOLDER in self.games.character_id:
            object.__setattr__(
                self,
                "games",
                self.games.model_copy(
                    update={"character_id": expand_character(self.games.character_id, self.id)}
                ),
            )
        return self

    @property
    def dir(self) -> Path:
        """``characters/<id>`` (relative to the CWD when built in code)."""
        return self._dir if self._dir is not None else Path("characters") / self.id

    def char_path(self, value: str) -> Path:
        """A file inside the character folder (persona, lexicon, filters overlay)."""
        path = Path(value)
        return path if path.is_absolute() else self.dir / path

    def resolve_path(self, value: str) -> Path:
        """A project-relative path (memory db, token file) with ``{character}`` expanded."""
        path = Path(expand_character(value, self.id))
        if path.is_absolute():
            return path
        return (self._root if self._root is not None else Path.cwd()) / path

    def persona_text(self) -> str:
        return self.char_path(self.persona).read_text(encoding="utf-8")

    def lexicon_map(self) -> dict[str, str]:
        return load_lexicon(self.char_path(self.lexicon))

    @property
    def emotion_map(self) -> dict[str, dict[str, Any]]:
        """``avatar.emotion_map`` as plain dicts, for the avatar's EmotionController."""
        return {k: v.model_dump(exclude_none=True) for k, v in self.avatar.emotion_map.items()}


def load_lexicon(path: Path) -> dict[str, str]:
    """Read a TTS lexicon (``[words]`` table or flat ``word = "reading"``). Missing → ``{}``."""
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return {}
    table = data.get("words", data)
    if not isinstance(table, Mapping):
        return {}
    return {str(k): str(v) for k, v in table.items() if isinstance(v, str)}


# --- secrets ----------------------------------------------------------------------------


class Secrets(BaseSettings):
    """API keys from ``.env`` / the environment. Never put these in TOML files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        frozen=True,
    )

    TYPHOON_API_KEY: SecretStr | None = None
    GEMINI_API_KEY: SecretStr | None = None
    AZURE_SPEECH_KEY: SecretStr | None = None
    AZURE_SPEECH_REGION: SecretStr | None = None
    YOUTUBE_API_KEY: SecretStr | None = None
    TWITCH_CLIENT_ID: SecretStr | None = None

    _extra: dict[str, str] = PrivateAttr(default_factory=dict)

    def get(self, name: str) -> str | None:
        """The plain value of secret ``name`` (a field, or any other key from .env/env)."""
        key = name.upper()
        if key in type(self).model_fields:
            value = getattr(self, key)
            return value.get_secret_value() if value is not None else None
        extra = self._extra.get(key)
        return extra or None

    def redaction_values(self) -> list[str]:
        """Secret values to scrub from logs (the Azure region is not secret)."""
        values = [self.get(n) for n in type(self).model_fields if n != "AZURE_SPEECH_REGION"]
        values += [v for k, v in self._extra.items() if is_secret_name(k)]
        return sorted({v for v in values if v})


def is_secret_name(name: str) -> bool:
    """Env names that hold credentials (API keys, tokens, passwords)."""
    upper = name.upper()
    return any(tag in upper for tag in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
