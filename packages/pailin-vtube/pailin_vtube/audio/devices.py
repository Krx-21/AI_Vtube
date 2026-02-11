"""
Audio device detection utilities.
"""

import logging
import platform
import shutil
import subprocess

logger = logging.getLogger(__name__)


def get_audio_devices() -> list[tuple[int, str]]:
    """
    Get a list of available audio output devices (cross-platform).

    Returns:
        List of (device_id, device_name) tuples.
    """
    devices = []
    system = platform.system()

    try:
        if system == "Windows":
            cmd = [
                "powershell",
                "-Command",
                "Get-CimInstance -ClassName Win32_SoundDevice | "
                "Where-Object { $_.Status -eq 'OK' } | "
                "Select-Object DeviceID, Name | Format-List",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            idx = 0
            for line in result.stdout.splitlines():
                if line.strip().startswith("Name"):
                    name = line.split(":", 1)[1].strip()
                    devices.append((idx, name))
                    idx += 1

        elif system == "Linux":
            if shutil.which("pactl"):
                result = subprocess.run(
                    ["pactl", "list", "short", "sinks"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for idx, line in enumerate(result.stdout.strip().splitlines()):
                    parts = line.split("\t")
                    name = parts[1] if len(parts) > 1 else parts[0]
                    devices.append((idx, name))

        elif system == "Darwin":
            # macOS: system_profiler SPAudioDataType outputs device names
            # as lines without a colon, while metadata lines use "key: value".
            # This heuristic may miss names containing colons.
            result = subprocess.run(
                ["system_profiler", "SPAudioDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            idx = 0
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if (
                    stripped
                    and not stripped.startswith(("Audio:", "Devices:"))
                    and ":" not in stripped
                ):
                    devices.append((idx, stripped))
                    idx += 1

    except Exception as e:
        logger.warning("Error getting audio devices: %s", e)

    return devices


def find_device(device_name: str, devices: list[tuple[int, str]] | None = None) -> int | None:
    """
    Find a device index matching *device_name* (case-insensitive).

    Args:
        device_name: Name of the device to find.
        devices: Optional list of devices. If not provided, will call get_audio_devices().

    Returns:
        Device index if found, None otherwise.
    """
    if not device_name:
        return None

    if devices is None:
        devices = get_audio_devices()

    lower = device_name.lower()

    # Exact match
    for idx, name in devices:
        if lower == name.lower():
            logger.info("Using audio device: %s (ID: %d)", name, idx)
            return idx

    # Partial match
    for idx, name in devices:
        if lower in name.lower():
            logger.info("Using audio device: %s (ID: %d)", name, idx)
            return idx

    # VB-Audio fallback
    if lower == "cable input":
        for idx, name in devices:
            if "vb-audio" in name.lower() or "virtual cable" in name.lower():
                logger.info("Using VB-Audio device: %s (ID: %d)", name, idx)
                return idx

    logger.warning(
        "Audio device '%s' not found. Using system default. Available: %s",
        device_name,
        [name for _, name in devices],
    )
    return None


def print_audio_devices() -> None:
    """Print all available audio output devices."""
    devices = get_audio_devices()
    print(f"Available audio output devices ({len(devices)}):")
    for idx, name in devices:
        print(f"  {idx}. {name}")


def print_device_setup_instructions() -> None:
    """Print instructions for routing audio through VB-Audio Virtual Cable."""
    print("How to set CABLE Input as your default audio device in Windows:")
    print("1. Right-click on the speaker icon in the system tray")
    print("2. Select 'Open Sound settings'")
    print("3. Under 'Output', select 'CABLE Input (VB-Audio Virtual Cable)'")
    print("4. Text-to-speech audio will now be routed through CABLE Input")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_audio_devices()
    print()
    print_device_setup_instructions()
