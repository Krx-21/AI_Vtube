# Windows audio I/O + barge-in — component brief

_Research snapshot: 2026-09-25. Verified facts carry a source; unverified items are marked._

## Recommendation

DEFAULT STACK for the RTX 4070 / i7-14700KF / Win11 PC (Python 3.12 recommended; 3.13 also verified):

1) I/O: python-sounddevice 0.5.6. Its Windows wheel bundles PortAudio v19.7.0. Always resolve devices through the "Windows WASAPI" host API, in shared mode with WasapiSettings(auto_convert=True). Never rely on sd.default: PortAudio's default host API on Windows is MME (about 90 ms latency, 31-char names). Never use exclusive mode on a streaming PC.
 - Output: a callback OutputStream that is opened once and always left running. 48000 Hz, float32, 2 ch, blocksize=480 (10 ms, the same as the Windows engine period), latency=0.04. It is fed from a lock-protected deque.
   - cancel() empties the queue under the lock. The next block contains an 8 ms fade, then silence. Time from cancel to silence is at most 1 block plus the device buffer (stream.latency, about 20-50 ms in WASAPI shared mode; estimated, measure it on the PC).
   - Levels (RMS, for lip-sync) and "segment finished" markers go out on a notifier thread. Each one is timed to outputBufferDacTime, the moment the audio is actually heard.
   - The output doubles as the exact far-end reference for AEC.
 - Mic: a callback InputStream, 48 kHz mono float32, 10 ms blocks. The callback only copies into a bounded queue. A consumer thread does all DSP.
2) Echo / barge-in default = mode "aec": WebRTC AEC3 through livekit 1.1.20 `rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True, high_pass_filter=True)`.
 - Why livekit: it is the only maintained AEC with ABI-independent wheels for win_amd64, manylinux and macOS, so it works on CI and on Python 3.11-3.14. It runs AEC3 at 48 kHz directly on mic frames, using the player's own output blocks as the reference. No reference resampling is needed because both run at 48 kHz with 10 ms blocks.
 - After AEC: soxr HQ 48k to 16k, then Silero VAD v6 ONNX through onnxruntime (no torch), then a segmenter.
 - Automatic fallback if livekit fails to import: mode "energy_dtd", a Geigel-style double-talk gate. The mic RMS must exceed k × learned coupling × the maximum reference RMS over the last 350 ms. No dependency.
 - Mode "half_duplex" (VAD ignored while the AI is speaking) stays as a config option for bad rooms, or when loud game audio also leaks.
 - Evidence (my synthetic benchmark: real speech, linear room echo with 40-300 ms bulk delay, no delay hint):
   - AEC3: ERLE 21-35 dB. VAD fired on echo alone 0-0.3% of frames. Streamer speech detected 85-96%. Barge-in about 0.4 s after the streamer starts talking. Cost 90 µs per 10 ms at 16 kHz, 137 µs at 48 kHz.
   - pyaec/speex: false-fires 10-64% without its preprocessor. With the preprocessor it loses 35-65% of streamer speech in some cases. It fails outright at 300 ms delay.
   - energy_dtd: 0-3% false-fire, but only 42-67% frame detection and barge-in about 0.6 s.
3) Orchestrator policy:
 - On speech_start{barge:true} while the AI is speaking: duck with player.gain = 0.35.
 - On barge_in (at least 250 ms of sustained speech): call cancel(). This is Neuro's "critical interrupts speech".
 - Queue a marker after each TTS sentence. Markers that fire True tell you what text the audience actually heard, so the LLM history can be cut at that point.
4) Routing:
 - Default: the AI plays on the streamer's default speakers (WASAPI default endpoint) and OBS "Desktop Audio" picks it up. Zero setup, and the AEC reference exactly matches what the speakers play.
 - Optional isolated track: a second StreamingPlayer on "CABLE Input" (VB-CABLE Pack45). In OBS add Audio Input Capture → "CABLE Output (VB-Audio Virtual Cable)". Do not also send the speaker copy into the same OBS track (the voice would be doubled). Keep the AEC reference on the speaker sink only.
 - OBS Application Audio Capture (BETA) works on OBS 28+ and Win10 2004+/Win11. It is selected by window: the capture follows the process tree of the window's owner. A console python.exe has no window of its own, so test it before relying on it.
5) Resampling: soxr 1.1.0 (fastest measured, HQ). Use a streaming ResampleStream per utterance and flush with last=True at each marker. Use one-shot soxr.resample for whole clips. samplerate 0.2.4 is the MIT fallback with a steady (non-bursty) stream. scipy resample_poly is stateless per chunk and only right for whole buffers.
6) Test on Linux CI with a dependency-injected fake backend. `import sounddevice` raises OSError on Linux without libportaudio2.

## Alternatives

### half_duplex (gate VAD/STT while the AI is speaking)
- **Pros:** Zero dependencies, zero false self-triggers, trivial to reason about. Also covers game or music leakage while she talks.
- **Cons:** The streamer cannot interrupt by voice, and words spoken over her are lost. 250 ms tail gating after she stops.
- **When:** Fallback for bad rooms or loud speakers, when livekit is unavailable and energy_dtd misfires, or when the streamer uses a hotkey or chat to interrupt (Neuro-style operator priority).

### energy_dtd (Geigel-style energy double-talk gate using the player reference; implemented)
- **Pros:** No dependency. Adapts to room coupling automatically while the AI is speaking. 0-3% false-fire in synthetic tests.
- **Cons:** Only 42-67% frame detection and about 0.6 s barge-in. Does not clean the audio, so STT still hears the echo mixed in. Weak when echo is as loud as the streamer or the speakers are nonlinear.
- **When:** Automatic fallback when `import livekit` fails.

### livekit rtc.AudioProcessingModule (WebRTC AEC3) - DEFAULT
- **Pros:** Best measured: 21-35 dB ERLE, about 0% false-fire, 85-96% detection, handles 40-300 ms delay without a hint. py3-none wheels for win_amd64, manylinux and macOS, so it works on CI and Python 3.9-3.14. Apache-2.0, maintained (release 2026-09-23). Adds NS and HPF.
- **Cons:** 10.7 MB wheel pulls protobuf and aiofiles. Goes through an FFI protobuf call per frame (~90-140 µs). Cosmetic shutdown assertion. Tied to LiveKit's release cadence.
- **When:** Default for a streamer without headphones.

### aec-audio-processing 1.0.1 (WebRTC APM v2 via SWIG)
- **Pros:** Same WebRTC APM family including VAD, bytes API, small (899 KB).
- **Cons:** Windows wheels only for cp311-313, no Linux wheels (CI needs swig and meson), no project URL, not runtime-tested here.
- **When:** Only if livekit's dependency footprint is unacceptable and CI is Windows-only.

### pyaec 1.0.1 (speexdsp MDF via aec-rs)
- **Pros:** 80 KB py3-none wheels on all platforms, MIT, simple API.
- **Cons:** 13-16 dB ERLE. Either leaks echo triggers (10-35%) or, with its preprocessor, suppresses the streamer. Fails when delay exceeds the filter tail. The ctypes list API is slow-ish.
- **When:** Emergency fallback; not recommended.

### Windows system AEC (Communications category, Voice Clarity or OEM APO)
- **Pros:** Zero CPU in our process. Microsoft AI AEC, NS and dereverb. Reference may include the full speaker mix (games too).
- **Cons:** Untested with PortAudio. Needs a private sounddevice field. Triggers -80% default ducking. Cannot choose the reference endpoint (IAcousticEchoCancellationControl is not exposed). Depends on driver or Windows build. Not available on Linux CI.
- **When:** A/B experiment on the user's PC (MicCapture(communications_mode=True), with livekit AEC off).

### Browser audio front-end (getUserMedia echoCancellation:true, TTS played in the same page; Open-LLM-VTuber style)
- **Pros:** Chrome's AEC3 for free, and the reference automatically covers audio Chrome plays.
- **Cons:** Moves audio I/O into a browser page (websocket PCM transport, extra latency, tab-throttling risk). Python core loses exact DAC timestamps.
- **When:** If the avatar or control UI is already a browser page and the Python path underperforms.

### WASAPI loopback full-mix reference (PyAudioWPatch 0.2.12.8, or soundcard 0.4.6 include_loopback)
- **Pros:** The AEC reference includes game, music and Discord audio from the speakers, so it cancels them too. PyAudioWPatch has cp38-cp314 win wheels.
- **Cons:** A second capture stream on another clock; extra dependency. Loopback is not available through the sounddevice bundled PortAudio 19.7.0. Untested here.
- **When:** Streamer plays games or music over speakers and gets false barge-ins from them.

### NVIDIA Maxine Audio Effects SDK AEC effect
- **Pros:** GPU AEC on RTX.
- **Cons:** Separate SDK and licensing; Python binding status unverified. The NVIDIA Broadcast app's 'Room Echo Removal' is dereverb, not speaker-echo cancellation.
- **When:** Not recommended for v1.

### PyAudio 0.2.14 / PyAudioWPatch instead of sounddevice
- **Pros:** Mature wheels. PyAudioWPatch adds WASAPI loopback.
- **Cons:** Bytes-based API with no numpy-native callbacks. PyAudio 0.2.14 has no cp314 wheel and no WasapiSettings equivalent for auto-convert.
- **When:** Only for the loopback reference stream.

### Resampling: samplerate 0.2.4 / scipy.signal.resample_poly
- **Pros:** samplerate: MIT, steady non-bursty streaming output. scipy: already a dependency for many projects.
- **Cons:** samplerate is 10x slower than soxr. scipy resample_poly is stateless (edge artifacts on chunks) and scipy 1.18 needs Python >=3.12.
- **When:** samplerate if LGPL (soxr) is a concern or burst-free streaming matters. scipy only for whole-buffer offline conversion.

### Routing: OBS Desktop Audio (default) vs Application Audio Capture vs VB-CABLE track
- **Pros:** Desktop Audio needs zero setup. App capture gives a separate fader without drivers (OBS 28+). VB-CABLE gives a deterministic, isolated device that VTube Studio mic-lipsync can also use.
- **Cons:** Desktop Audio mixes everything. App capture is BETA and needs a window-owner process tree. VB-CABLE adds a driver install, can take over the default devices, and has up to ~149 ms buffer. Capturing both the speaker and cable/app copies doubles the voice.
- **When:** Desktop Audio for v1. VB-CABLE second sink when a separate VOD or mix track is wanted.

## Verified facts

- ✅ sounddevice 0.5.6 released 2026-08-17. Files: py3-none-any, macOS universal2, py3-none-win32, py3-none-win_amd64, py3-none-win_arm64, and an sdist. The Windows wheels are pure Python plus DLLs, so they work on any CPython >=3.7, including 3.11-3.13. Only dependency: cffi (numpy is an extra).  
  Source: https://pypi.org/pypi/sounddevice/json ; https://raw.githubusercontent.com/spatialaudio/python-sounddevice/master/NEWS.rst
- ✅ The sounddevice Windows wheel bundles PortAudio v19.7.0: _sounddevice_data/portaudio-binaries/libportaudio64bit.dll plus an -asio variant, chosen with the SD_ENABLE_ASIO env var (since 0.5.1). The DLL contains the string 'PortAudio V19.7.0-devel'. Host APIs in the default DLL: MME, DirectSound, WDM-KS, WASAPI.  
  Source: Inspected sounddevice-0.5.6-py3-none-win_amd64.whl; its build-libs.yml checks out PortAudio ref v19.7.0; https://raw.githubusercontent.com/spatialaudio/portaudio-binaries/master/README.md
- ✅ Signature is sd.WasapiSettings(exclusive=False, auto_convert=False, explicit_sample_format=False). auto_convert was added in 0.4.7 and maps to paWinWasapiAutoConvert. explicit_sample_format was added in 0.5.3.  
  Source: sounddevice.py 0.5.6 source (class WasapiSettings); NEWS.rst
- ✅ With paWinWasapiAutoConvert in shared mode, PortAudio 19.7.0 sets AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY and reports any rate or channel format as supported, so Windows does the SRC and remixing. Without the flag, a rate that differs from the endpoint mix format fails to open.  
  Source: https://raw.githubusercontent.com/PortAudio/portaudio/v19.7.0/src/hostapi/wasapi/pa_win_wasapi.c (lines ~2925, ~3848, ~3984)
- ✅ In WASAPI shared mode PortAudio 19.7.0 clamps the host period to at least IAudioClient DefaultDevicePeriod (typically 10 ms). defaultLow/HighOutputLatency for WASAPI devices come from MinimumDevicePeriod and DefaultDevicePeriod.  
  Source: pa_win_wasapi.c v19.7.0 lines ~2043-2053, ~3368-3376
- ✅ PortAudio default latencies: MME on WDM is 0.090 s low (PA_MME_WIN_WDM_DEFAULT_LATENCY_). DirectSound on WDM is 0.120 s (PA_DS_WIN_WDM_DEFAULT_LATENCY_).  
  Source: https://raw.githubusercontent.com/PortAudio/portaudio/v19.7.0/src/hostapi/wmme/pa_win_wmme.c ; .../dsound/pa_win_ds.c
- ✅ PortAudio's default host API on Windows is the first initialiser that has devices, which is MME. So sd.default.device and devices without an explicit host API resolve to MME.  
  Source: https://raw.githubusercontent.com/PortAudio/portaudio/v19.7.0/src/os/win/pa_win_hostapis.c ; src/common/pa_front.c
- ✅ Windows host API names are exactly 'MME', 'Windows DirectSound', 'Windows WASAPI' and 'Windows WDM-KS'. MME device names come from WAVEOUTCAPS.szPname (32 chars including NUL), so they are truncated to 31 chars.  
  Source: pa_win_wasapi.c / pa_win_wmme.c / pa_win_ds.c / pa_win_wdmks.c v19.7.0
- ✅ sounddevice device-string matching is case-insensitive: space-separated substrings must appear in order within 'device name, host API name'. More than one match raises ValueError unless exactly one is an exact name match.  
  Source: sounddevice.py 0.5.6 _get_device_id()
- ✅ Callback contract: callback(outdata, frames, time, status). time.inputBufferAdcTime, time.outputBufferDacTime and time.currentTime share the stream clock. If the callback raises, it is never called again. The callback must always fill outdata.  
  Source: sounddevice.py 0.5.6 docstrings (Stream callback section)
- ✅ PortAudio 19.7.0 WASAPI: when a processing-thread error occurs (for example the device is invalidated), the thread runs _StreamOnStop, which calls streamFinishedCallback and sets running=FALSE (stream.active becomes False). Disconnects can therefore be detected through finished_callback or stream.active. This conclusion comes from reading the code path; I did not test it on real hardware.  
  Source: pa_win_wasapi.c v19.7.0 lines ~5770-5784, ~6000-6021, IsStreamActive ~4594
- ✅ The PortAudio device list is frozen at Pa_Initialize. PaWasapi_UpdateDeviceList() only exists if PA_WASAPI_MAX_CONST_DEVICE_COUNT>0 at compile time (the default is 0). The community workaround is sd._terminate(); sd._ffi.dlclose(sd._lib); sd._lib=sd._ffi.dlopen(sd._libname); sd._initialize(). It is a private API and invalidates all streams.  
  Source: pa_win_wasapi.h v19.7.0; https://github.com/spatialaudio/python-sounddevice/issues/516 ; https://github.com/spatialaudio/python-sounddevice/issues/47
- ✅ PaWasapiStreamInfo in PortAudio >=19.6 has streamCategory (eAudioCategoryCommunications=3) and streamOption (Raw=1, MatchFormat=2). PortAudio applies them via IAudioClient2::SetClientProperties. The sounddevice cffi cdef exposes these fields, so they can be set through WasapiSettings()._streaminfo.streamCategory (private API).  
  Source: pa_win_wasapi.h/.c v19.7.0 (~1750-1778); sounddevice_build.py on master
- ✅ On Linux without libportaudio2, `import sounddevice` raises OSError('PortAudio library not found'). After `apt-get install libportaudio2` (Ubuntu 24.04 ships 19.6.0) it imports, but a headless container lists 0 devices.  
  Source: Tested in this container (CPython 3.11.15)
- ✅ soxr 1.1.0 (2026-05-03), license LGPL-2.1-or-later. Wheels: cp39-cp311, cp312-abi3 (covers 3.12/3.13/3.14) and cp314t, for win_amd64, manylinux and macOS. API: soxr.resample(x, in_rate, out_rate, quality='HQ') and soxr.ResampleStream(in_rate, out_rate, num_channels, dtype='float32', quality='HQ', vr=False) with .resample_chunk(x, last=False), .clear(), .delay(), .num_clips(), .set_io_ratio().  
  Source: https://pypi.org/pypi/soxr/json ; inspected in the installed package
- ✅ soxr ResampleStream output is bursty for small chunks. 48k to 16k with 10 ms input: HQ gives ~490 samples every 3rd block (first output after ~40 ms); LQ gives 300 samples every other block; QQ gives a steady 160. Cost is about 3-6 µs per 10 ms block. Output is time-aligned (impulse lands at the correct sample). The tail must be flushed with last=True.  
  Source: Measured in this container
- ✅ samplerate 0.2.4 (2026-03-22), MIT, wheels cp39-cp314 win_amd64. Its streaming Resampler gives a steady 160 out per 480 in. Cost per 10 ms block: sinc_fastest 37 µs, sinc_medium 79 µs, sinc_best 250 µs. scipy resample_poly on a 10 ms block (stateless) costs 125 µs.  
  Source: https://pypi.org/pypi/samplerate/json ; measured in this container
- ✅ numpy 2.5.3 and scipy 1.18.1 require Python >=3.12. The newest releases with cp311 win_amd64 wheels are numpy 2.4.6 and scipy 1.17.1. onnxruntime 1.30.0 requires Python >=3.11 (cp311-cp314 win_amd64).  
  Source: https://pypi.org/pypi/numpy/json ; https://pypi.org/pypi/scipy/json ; https://pypi.org/pypi/onnxruntime/json
- ✅ livekit 1.1.20 (2026-09-23), Apache-2.0. Wheels are py3-none for win_amd64 (10.7 MB), manylinux_2_28 x86_64/aarch64 and macOS. Dependencies: protobuf>=5, types-protobuf>=5, aiofiles>=24, numpy>=1.26. API: rtc.AudioProcessingModule(echo_cancellation, noise_suppression, high_pass_filter, auto_gain_control), with process_stream(AudioFrame), process_reverse_stream(AudioFrame) and set_stream_delay_ms(int). Frames must be exactly 10 ms int16 and are modified in place. rtc.AudioFrame(data, sample_rate, num_channels, samples_per_channel).  
  Source: https://pypi.org/pypi/livekit/json ; https://raw.githubusercontent.com/livekit/python-sdks/main/livekit-rtc/livekit/rtc/apm.py ; audio_frame.py ; used successfully at 16 and 48 kHz in this container
- ✅ livekit's APM is libwebrtc AEC3: webrtc-sys/include/livekit/apm.h includes modules/audio_processing/aec3/echo_canceller3.h and sets config.echo_canceller.enabled, gain_controller2, high_pass_filter and noise_suppression.  
  Source: https://raw.githubusercontent.com/livekit/rust-sdks/main/webrtc-sys/include/livekit/apm.h ; webrtc-sys/src/apm.cpp
- ✅ aec-audio-processing 1.0.1 (2025-09-01), BSD-3-Clause. It ships only cp311/cp312/cp313 win_amd64 wheels plus an sdist (building needs swig and meson): no Linux wheels, no cp314. It bundles webrtc-audio-processing-2-1.dll (the PulseAudio fork of WebRTC APM v2). API: AudioProcessor(enable_aec=True, enable_ns=True, ns_level=2, enable_agc=True, agc_mode=1, enable_vad=True), .set_stream_format(sr_in, ch_in, sr_out, ch_out), .set_reverse_stream_format(sr, ch), .set_stream_delay(ms), .process_stream(bytes) -> bytes, .process_reverse_stream(bytes), .has_voice(). I inspected the wheel but could not run it (Windows-only).  
  Source: https://pypi.org/pypi/aec-audio-processing/json ; inspected aec_audio_processing-1.0.1-cp312-cp312-win_amd64.whl
- ✅ pyaec 1.0.1 (2024-12-08) is ctypes bindings to thewh1teagle/aec-rs (speexdsp MDF echo canceller, MIT). py3-none wheels for win_amd64, win_arm64, manylinux x86_64/aarch64 and macOS. API: pyaec.Aec(frame_size, filter_length, sample_rate, enable_preprocess).cancel_echo(rec_list_int16, echo_list_int16) -> list. Cost is about 100-120 µs per 10 ms at 16 kHz.  
  Source: https://pypi.org/pypi/pyaec/json ; inspected wheel pyaec/__init__.py ; https://raw.githubusercontent.com/thewh1teagle/aec-rs/main/README.md ; measured
- ✅ Ruled out as Windows AEC options: webrtc-noise-gain 1.3.0 (noise suppression and AGC only, no AEC, manylinux wheels only); speexdsp 0.1.1 (2018, sdist only); webrtc-audio-processing 0.1.3 (2019, armv7 wheels only); webrtc-apm 0.1.6 (a single cp311 win wheel, placeholder homepage).  
  Source: https://pypi.org/pypi/<name>/json for each package
- ✅ The silero-vad 6.2.3 pip package (2026-09-23) requires torch, and utils_vad.py imports torch at module level. The ONNX model (src/silero_vad/data/silero_vad.onnx, 2,327,524 B, sha256 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3) runs with onnxruntime alone. Inputs: input [1, 64+512] float32, state [2,1,128], sr int64. Outputs: output [1,1] and stateN. At 16 kHz the window is 512 samples (32 ms). Cost was about 0.15 ms per window.  
  Source: https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/utils_vad.py ; https://pypi.org/pypi/silero-vad/json ; tested in this container
- ✅ OBS Application Audio Capture (BETA): available since OBS 28 on Windows 10 2004+ and Windows 11. OBS 30.1 added a 'Capture Audio (BETA)' checkbox on Window and Game Capture. It is still labelled BETA. OBS recommends matching by executable for apps whose window titles change.  
  Source: https://obsproject.com/kb/application-audio-capture-guide
- ✅ OBS win-wasapi process capture uses AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK with PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE. The target is the PID from GetWindowThreadProcessId(hwnd) of the selected window, so it captures that window-owner process and its descendants.  
  Source: https://raw.githubusercontent.com/obsproject/obs-studio/master/plugins/win-wasapi/win-wasapi.cpp (lines ~696-709, ~867)
- ✅ VB-CABLE: current package VBCABLE_Driver_Pack45.zip (Oct 2024), donationware, XP to Win11 32/64-bit and Arm64. Install: extract, run VBCABLE_Setup_x64.exe as administrator, reboot. Endpoints are CABLE Input (playback) and CABLE Output (recording). Driver 3.3.1.7. Internal SR 48000 Hz. Default Max Latency and Latency are 7168 samples. On Win10/11 it installs 2 playback inputs: an 8-ch speaker pin and a 16-ch Line Out pin. Windows may make it the default playback and recording device after install.  
  Source: https://vb-audio.com/Cable/ ; https://vb-audio.com/Cable/VBCABLE_ReferenceManual.pdf (Oct 2024, rev 3)
- ⚠️ unverified — The full endpoint names 'CABLE Input (VB-Audio Virtual Cable)' and 'CABLE Output (VB-Audio Virtual Cable)' come from third-party guides. The manual gives the driver name as 'VB-Audio Virtual Cable'. Match on the substring 'CABLE Input'.  
  Source: Web search results (obsproject forum, voxbooster) + VB manual
- ✅ Windows default stream attenuation: when a communications stream opens, Windows attenuates other streams. The default is 'Reduce the volume of other sounds by 80%' (Sound control panel > Communications tab).  
  Source: https://learn.microsoft.com/en-us/windows/win32/coreaudio/stream-attenuation
- ✅ IAcousticEchoCancellationControl::SetEchoCancellationRenderEndpoint (Windows build 22621+) selects the render endpoint used as the system AEC reference. PortAudio and sounddevice do not expose it.  
  Source: https://learn.microsoft.com/en-us/windows/win32/api/audioclient/nn-audioclient-iacousticechocancellationcontrol
- ⚠️ unverified — Windows 11 Voice Clarity is an AI echo cancellation, noise suppression and dereverb feature applied only to apps using the Communications signal-processing mode. It runs on x64 and Arm64 without an NPU. My sources were secondary (news and blog posts); I did not find a primary Microsoft developer page. It is untested whether a PortAudio WASAPI stream tagged Communications receives it.  
  Source: https://www.neowin.net/news/one-of-windows-11-ai-features-is-now-available-to-all-users-no-special-hardware-required/ ; https://learn.microsoft.com/en-us/windows-hardware/test/hlk/testref/voice-clarity-system-verification-test-loopback-sample
- ✅ PyAudioWPatch 0.2.12.8 (2026-01-14) is a PyAudio fork with WASAPI loopback: speakers appear as extra input devices. Wheels cp38-cp314 win32/win_amd64. It can serve as a full-mix AEC reference when game audio also plays on the speakers.  
  Source: https://pypi.org/pypi/pyaudiowpatch/json
- ✅ Synthetic benchmark (real speech from silero test.wav as the AI voice and JFK as the streamer; linear RIR, RT60 0.3 s; bulk delay 40/150/300 ms; no delay hint). Results:
- livekit AEC3: ERLE 21-35 dB; VAD false-fire on echo alone 0-0.3%; streamer speech detected 85-96%.
- pyaec speex (tail 4096): ERLE 13-16 dB with false-fire 10-35%. At 300 ms delay it fails completely (1 dB ERLE, 64% false-fire).
- Energy double-talk gate on the raw mic: false-fire 0-3%, detection 42-67%.
- End-to-end VoiceFrontEnd with 120 ms delay, streamer starting at ~4.35 s: aec mode gave speech_start 4.65 s and barge_in 4.74 s; energy_dtd gave 4.83 s and 4.96 s; half_duplex gave no events. No false events in 0-4 s.
The signals were linear and synthetic, with no loudspeaker nonlinearity or clock drift.  
  Source: audio/aec_bench2.py and audio/test_audio_io.py run in this container
- ⚠️ unverified — Open-LLM-VTuber (a clone stack, not Neuro) advertises voice interruption without headphones using the browser's echo cancellation in its web frontend.  
  Source: https://github.com/Open-LLM-VTuber/Open-LLM-VTuber ; https://dev.to/andrew-ooo/open-llm-vtuber-review-offline-ai-companion-with-live2d-327m

## Install (Windows)

```
REM Python 3.12 x64 (numpy 2.5 and scipy 1.18 need >=3.12; 3.11 works with numpy<=2.4.6 and scipy<=1.17.1)
winget install -e --id Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install "sounddevice==0.5.6" "soxr==1.1.0" "numpy>=2.1,<3" "onnxruntime==1.30.0" "livekit==1.1.20"
REM optional: speex fallback, MIT resampler, WASAPI loopback (full-mix reference), test deps
pip install "pyaec==1.0.1" "samplerate==0.2.4" "PyAudioWPatch==0.2.12.8" soundfile pytest
REM Silero VAD model (use raw.githubusercontent.com; do NOT pip install silero-vad, it pulls torch)
mkdir models
curl -L -o models\silero_vad.onnx https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
certutil -hashfile models\silero_vad.onnx SHA256   REM expect 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3 (master as of 2026-09-25)
REM sanity check: WASAPI devices and PortAudio version
python -c "import sounddevice as sd; print(sd.get_portaudio_version()); print(sd.query_hostapis()); print(sd.query_devices())"
python audio_demo_windows.py --list
python audio_demo_windows.py --out "Speakers" --mic "Microphone" --mode aec --vad models\silero_vad.onnx

REM VB-CABLE (optional, only for an isolated OBS track):
REM   1. download VBCABLE_Driver_Pack45.zip from https://vb-audio.com/Cable/ and extract to a local folder
REM   2. right-click VBCABLE_Setup_x64.exe > Run as administrator > Install Driver
REM   3. reboot
REM   4. Settings > System > Sound: RESET default output/input to your real speakers/mic (VB-CABLE may take over as default)
REM   5. optional: lower VBCABLE_ControlPanel "Latency" from 7168 samples (~149 ms max buffer at 48 kHz) if you need a tighter buffer; keep Internal SR 48000
REM OBS (>=28):
REM   default route: nothing to do; "Desktop Audio" already carries the AI voice played on the default speakers.
REM   cable route: Sources + > Audio Input Capture > Device "CABLE Output (VB-Audio Virtual Cable)"; Advanced Audio Properties: pick its track and Monitor Off (you already hear the speaker copy).
REM   per-app route: Sources + > Application Audio Capture (BETA) > select the window, Priority "Match executable"; needs a window owned by (or an ancestor of) the python process.
REM Windows (only if the experimental communications-mode capture is enabled): Control Panel > Sound > Communications tab > "Do nothing" (default is -80% ducking of other sounds).
REM CI: ubuntu tests inject FakeSD, so no apt package is needed; if a job must import sounddevice on Linux: sudo apt-get install -y libportaudio2
```

## API notes

FILES. Tested: 7/7 pytest pass on CPython 3.11.15 and 3.13.12 in the Linux container with a fake PortAudio backend. Also smoke-tested against real sounddevice 0.5.6 on an ALSA null device.
  /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/audio/audio_io.py (module below, verbatim)
  /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/audio/test_audio_io.py
    FakeSD/FakeStream: query_hostapis, query_devices, WasapiSettings, OutputStream, InputStream; .pump(n) drives output callbacks; .push(x) drives input callbacks; .die() fires finished_callback.
  /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/audio/audio_demo_windows.py (hardware demo for the streaming PC; NOT runnable in the container)
  /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/audio/aec_bench2.py (AEC and barge-in benchmark; needs sp_test.wav, sp_jfk.wav, silero_vad.onnx in the same folder)

CORE RULES
- Resolve devices yourself from sd.query_hostapis() and sd.query_devices() filtered to 'Windows WASAPI'. Pass the int index; do not pass strings to sounddevice (MME, DS and WDM-KS duplicates cause "Multiple devices found").
- With query None, use query_hostapis()[wasapi]['default_output_device'].
- Pass extra_settings=sd.WasapiSettings(auto_convert=True) only to WASAPI devices (other host APIs reject it).
- Output: sd.OutputStream(device=idx, samplerate=48000, channels=2, dtype='float32', blocksize=480, latency=0.04, extra_settings=..., callback=cb, finished_callback=fin). Keep it running forever and write zeros when idle, so there is no open/start latency per utterance.
- Callback: take a threading.Lock (hold time is microseconds), copy from the deque, compute RMS, append to an events deque. No allocation-heavy work, no I/O, no PortAudio calls. Wrap everything in try/except and zero outdata on error. Never call stream.close/abort inside finished_callback; only set an Event.
- Audible time = time.monotonic() + (time_info.outputBufferDacTime - time_info.currentTime). MME/DS may report 0; fall back to stream.latency.
- Capture time = now - (currentTime - inputBufferAdcTime).
- Cancel: under the lock, take ~8 ms of the head audio times a 1→0 ramp as the fade, clear the deque, fire pending markers with False. The next callback plays the fade and then zeros.
- Markers: an in-band _Marker item in the deque. The callback turns it into an event timestamped at the audible time of its sample position. Use one marker per TTS sentence plus a final one (like the Neuro SDK's segments + final marker).
- Lip-sync: on_level(rms) from the notifier thread at <=60 Hz, peak-held between reports. Map to a VTube Studio MouthOpen value in 0..1, for example clip((20*log10(rms+1e-9)+50)/30, 0, 1). Call from the notifier thread via loop.call_soon_threadsafe (do not touch asyncio objects directly).
- AEC3 (livekit):
  - Per 10 ms: apm.process_reverse_stream(rtc.AudioFrame(ref_int16.tobytes(), 48000, 1, 480)) for every reference block available, then apm.set_stream_delay_ms(0), then apm.process_stream(mic_frame). The output is in mic_frame.data (memoryview of int16), in place.
  - The render (reference) must be fed before its echo is captured. Feeding the player tap at callback time guarantees this, because the tap leads the audible time by the output latency.
  - The AEC3 delay estimator handled 40-300 ms with hint 0 in my tests.
- Silero ONNX (no torch):
  - x = concat(ctx64, window512)
  - out, state = sess.run(None, {'input': x[None], 'state': state, 'sr': np.array(16000, np.int64)})
  - ctx64 = x[-64:]
- asyncio bridge: on_event = lambda k,t,p: loop.call_soon_threadsafe(evq.put_nowait, (k,t,p)). Do the same for on_level and marker callbacks.
- Two sinks (speakers + CABLE): create two StreamingPlayer instances and call play/mark/cancel on both. Only the speaker player's .reference feeds the AEC. Set the same gain on both.
- Experimental Windows system AEC: MicCapture(..., communications_mode=True) sets WasapiSettings()._streaminfo.streamCategory = sd._lib.eAudioCategoryCommunications (private API). It may trigger Voice Clarity and triggers Windows ducking. It is untested. Use it only as an A/B experiment against livekit AEC3; never both at once.

audio_io.py (verbatim, 608 lines):

"""audio_io.py - low-latency audio I/O for a voice agent (Windows 11 first, testable on Linux).

Runtime deps : numpy, sounddevice>=0.5.6 (Windows wheel bundles PortAudio 19.7.0), soxr>=1.0
Optional     : onnxruntime (Silero VAD ONNX), livekit>=1.1 (WebRTC AEC3 via rtc.AudioProcessingModule)

Every PortAudio call goes through `backend` (defaults to the real `sounddevice` module) so unit
tests inject a fake: on Linux without libportaudio2, `import sounddevice` raises
OSError('PortAudio library not found').
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger("audio_io")

# PortAudio's *global* default host API on Windows is MME (first initialiser in pa_win_hostapis.c),
# which means ~90 ms+ latency and 31-char truncated names. Always resolve devices through WASAPI first.
HOSTAPI_PREF = ("Windows WASAPI", "MME", "Windows DirectSound")


def _real_sd():
    import sounddevice as sd  # lazy: raises OSError on Linux without libportaudio2
    return sd


def _safe(cb: Optional[Callable], *a: Any) -> None:
    if cb is None:
        return
    try:
        cb(*a)
    except Exception:  # never let a user callback kill an audio/notifier thread
        log.exception("audio_io callback failed")


# ----------------------------------------------------------------------------- devices
def list_devices(backend=None) -> list[dict]:
    sd = backend or _real_sd()
    apis = sd.query_hostapis()
    return [
        dict(index=i, name=d["name"], hostapi=apis[d["hostapi"]]["name"],
             max_in=d["max_input_channels"], max_out=d["max_output_channels"],
             default_sr=d["default_samplerate"])
        for i, d in enumerate(sd.query_devices())
    ]


def resolve_device(query: "str | int | None", kind: str, backend=None, hostapis=HOSTAPI_PREF) -> int:
    """None -> the WASAPI default endpoint (not PortAudio's MME default).
    str  -> case-insensitive substring of the device name (e.g. "CABLE Input", "Realtek"),
            first match in host-API preference order.  int -> used as-is."""
    sd = backend or _real_sd()
    if isinstance(query, int):
        return query
    apis = list(sd.query_hostapis())
    key = "max_out" if kind == "output" else "max_in"
    devs = list_devices(sd)
    for api_name in hostapis:
        api_idx = next((i for i, a in enumerate(apis) if a["name"] == api_name), None)
        if api_idx is None:
            continue
        if query is None:
            idx = apis[api_idx]["default_output_device" if kind == "output" else "default_input_device"]
            if idx is not None and idx >= 0:
                return idx
            continue
        q = query.lower()
        for d in devs:
            if d["hostapi"] == api_name and d[key] > 0 and q in d["name"].lower():
                return d["index"]
    raise LookupError(f"no {kind} device matching {query!r} in host APIs {hostapis}")


def _extra_settings(sd, device_index: int, communications: bool = False):
    api = sd.query_hostapis(sd.query_devices(device_index)["hostapi"])["name"]
    if api != "Windows WASAPI":
        return None
    # auto_convert = paWinWasapiAutoConvert: lets Windows resample/remix when our rate/channels differ
    # from the endpoint mix format (shared mode). Never use exclusive=True on a streaming PC.
    s = sd.WasapiSettings(auto_convert=True)
    if communications:  # EXPERIMENTAL: tag stream AudioCategory_Communications (PortAudio>=19.6 field);
        s._streaminfo.streamCategory = sd._lib.eAudioCategoryCommunications  # private sounddevice API
    return s


def reinit_portaudio(backend=None) -> None:
    """Refresh PortAudio's frozen device list (hot-plug). CLOSES/INVALIDATES ALL STREAMS in the process.
    Private sounddevice API; dlclose/dlopen per python-sounddevice issue #516."""
    sd = backend or _real_sd()
    sd._terminate()
    try:
        sd._ffi.dlclose(sd._lib)
        sd._lib = sd._ffi.dlopen(sd._libname)
    except Exception:
        log.warning("PortAudio DLL reload failed; plain re-init only", exc_info=True)
    sd._initialize()


# ----------------------------------------------------------------------------- player
@dataclass
class _Chunk:
    pcm: np.ndarray            # float32 mono @ device rate
    pos: int = 0


@dataclass
class _Marker:
    cb: Callable[[bool], None]  # cb(True) = everything before it was heard; cb(False) = cancelled


class StreamingPlayer:
    """Always-open callback OutputStream fed from a lock-protected deque.

    play(pcm, sr)  float32 mono chunks at any rate (resampled with soxr, stream state kept per utterance)
    mark(cb)       cb(True) when all audio queued before it has been *heard* (DAC-time aligned)
    cancel()       flush within one block (+ what is already inside the device buffer), short fade-out
    on_level(rms)  called on a notifier thread at <= level_hz, aligned to audible time (lip-sync)
    reference      deque[(t_audible_monotonic, block_f32)] - far-end reference for AEC / double-talk
    """

    def __init__(self, device: "str | int | None" = None, *, samplerate: int = 48000, channels: int = 2,
                 blocksize: int = 480, latency: "float | str" = 0.04, on_level: Optional[Callable[[float], None]] = None,
                 level_hz: float = 60.0, fade_ms: float = 8.0, on_device_lost: Optional[Callable[[], None]] = None,
                 backend=None):
        self._sd = backend or _real_sd()
        self.device_query, self.samplerate, self.channels = device, samplerate, channels
        self.blocksize, self.latency, self.on_level, self.level_hz = blocksize, latency, on_level, level_hz
        self.on_device_lost = on_device_lost
        self.gain = 1.0
        self.reference: collections.deque = collections.deque(maxlen=int(3 * samplerate / blocksize))
        self.stats = collections.Counter()
        self._q: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._rs_lock = threading.Lock()
        self._rs = None
        self._rs_sr = None
        self._fade: Optional[np.ndarray] = None
        self._fade_len = max(1, int(samplerate * fade_ms / 1000))
        self._mix = np.zeros(blocksize * 8, np.float32)
        self._events: collections.deque = collections.deque()
        self._speaking_until = 0.0
        self._out_latency = 0.05
        self._stream = None
        self._closing = False
        self._lost = threading.Event()
        self._stop = threading.Event()
        self._last_cb = time.monotonic()
        self._thr: Optional[threading.Thread] = None

    # -- lifecycle
    def start(self) -> "StreamingPlayer":
        self._open()
        self._thr = threading.Thread(target=self._notifier, name="player-notify", daemon=True)
        self._thr.start()
        return self

    def close(self) -> None:
        self._closing = True
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
        if self._thr:
            self._thr.join(timeout=1)

    def _open(self) -> None:
        sd = self._sd
        idx = resolve_device(self.device_query, "output", sd)
        self._stream = sd.OutputStream(
            device=idx, samplerate=self.samplerate, channels=self.channels, dtype="float32",
            blocksize=self.blocksize, latency=self.latency, extra_settings=_extra_settings(sd, idx),
            callback=self._callback, finished_callback=self._on_finished)
        self._stream.start()
        self._out_latency = float(self._stream.latency)
        self._last_cb = time.monotonic()
        self._lost.clear()
        log.info("output device %s opened, latency %.1f ms", idx, self._out_latency * 1e3)

    def _on_finished(self) -> None:  # PortAudio thread: never call PortAudio from here
        if not self._closing:
            self._lost.set()

    # -- producer API (any thread)
    def play(self, pcm: np.ndarray, sr: int) -> None:
        x = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
        if sr != self.samplerate:
            import soxr
            with self._rs_lock:
                if self._rs is None or self._rs_sr != sr:
                    self._flush_resampler_locked()
                    self._rs = soxr.ResampleStream(sr, self.samplerate, 1, dtype="float32", quality="HQ")
                    self._rs_sr = sr
                x = self._rs.resample_chunk(x)
        if x.size:
            with self._lock:
                self._q.append(_Chunk(x))

    def mark(self, cb: Callable[[bool], None]) -> None:
        with self._rs_lock:
            self._flush_resampler_locked()   # soxr holds ~tens of ms internally; push it out first
        with self._lock:
            self._q.append(_Marker(cb))

    def _flush_resampler_locked(self) -> None:
        if self._rs is not None:
            tail = self._rs.resample_chunk(np.zeros(0, np.float32), last=True)
            self._rs = None
            if tail.size:
                with self._lock:
                    self._q.append(_Chunk(np.ascontiguousarray(tail, dtype=np.float32)))

    def cancel(self) -> float:
        """Drop everything queued; returns seconds of audio dropped. Pending markers get cb(False)."""
        with self._rs_lock:
            self._rs = None
        with self._lock:
            need, tail, dropped, markers = self._fade_len, [], 0, []
            for it in self._q:
                if isinstance(it, _Marker):
                    markers.append(it)
                    continue
                rem = it.pcm[it.pos:]
                dropped += rem.size
                if need > 0:
                    tail.append(rem[:need])
                    need -= min(need, rem.size)
            self._q.clear()
            if tail:
                f = np.concatenate(tail)
                self._fade = f * np.linspace(1.0, 0.0, f.size, dtype=np.float32)
        now = time.monotonic()
        self._speaking_until = min(self._speaking_until, now + self._out_latency + self._fade_len / self.samplerate)
        for m in markers:
            self._events.append(("mark", now, partial(m.cb, False)))
        self.stats["cancel"] += 1
        return dropped / self.samplerate

    def is_speaking(self, tail_s: float = 0.25) -> bool:
        with self._lock:
            pending = any(isinstance(i, _Chunk) for i in self._q) or self._fade is not None
        return pending or time.monotonic() < self._speaking_until + tail_s

    def ref_rms_max(self, t0: float, t1: float) -> float:
        """Max block RMS of what was audible in [t0, t1] (monotonic). For energy double-talk detection."""
        m = 0.0
        for t, blk in reversed(self.reference):
            if t < t0 - 0.05:
                break
            if t <= t1:
                m = max(m, float(np.sqrt(np.dot(blk, blk) / blk.size)))
        return m

    # -- audio thread
    def _callback(self, outdata, frames, time_info, status) -> None:
        try:
            if status.output_underflow:
                self.stats["underflow"] += 1
            if frames > self._mix.size:
                self._mix = np.zeros(frames, np.float32)
            buf = self._mix[:frames]
            buf.fill(0.0)
            n = 0
            now = time.monotonic()
            dac, cur = time_info.outputBufferDacTime, time_info.currentTime
            delay = (dac - cur) if (dac > 0 and dac >= cur) else self._out_latency
            t0 = now + delay
            with self._lock:
                if self._fade is not None:
                    k = min(frames, self._fade.size)
                    buf[:k] = self._fade[:k]
                    self._fade = self._fade[k:] if k < self._fade.size else None
                    n = frames                         # rest of this block stays silent
                while n < frames and self._q:
                    it = self._q[0]
                    if isinstance(it, _Marker):
                        self._events.append(("mark", t0 + n / self.samplerate, partial(it.cb, True)))
                        self._q.popleft()
                        continue
                    take = min(frames - n, it.pcm.size - it.pos)
                    buf[n:n + take] = it.pcm[it.pos:it.pos + take]
                    it.pos += take
                    n += take
                    if it.pos >= it.pcm.size:
                        self._q.popleft()
                if self._q and isinstance(self._q[0], _Marker):  # marker right at block end
                    self._events.append(("mark", t0 + frames / self.samplerate, partial(self._q.popleft().cb, True)))
            if self.gain != 1.0:
                buf *= self.gain
            outdata[:] = buf[:, None]
            rms = float(np.sqrt(np.dot(buf, buf) / frames))
            self._events.append(("lvl", t0, rms))
            if rms > 1e-4:
                self._speaking_until = t0 + frames / self.samplerate
            self.reference.append((t0, buf.copy()))
            self._last_cb = now
        except Exception:  # an exception would stop PortAudio from ever calling us again
            outdata.fill(0)
            self.stats["cb_error"] += 1

    # -- notifier / watchdog thread
    def _notifier(self) -> None:
        peak, have, next_lvl, next_try = 0.0, False, 0.0, 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            ev = self._events
            while ev and ev[0][1] <= now:
                kind, _, x = ev.popleft()
                if kind == "lvl":
                    peak, have = max(peak, x), True
                else:
                    _safe(x)
            if have and self.on_level and now >= next_lvl:
                _safe(self.on_level, peak)
                peak, have, next_lvl = 0.0, False, now + 1.0 / self.level_hz
            stalled = self._stream is not None and now - self._last_cb > 1.0
            if (self._lost.is_set() or stalled) and now >= next_try:
                self.stats["device_lost"] += 1
                log.warning("output stream lost (finished=%s stalled=%s); reopening", self._lost.is_set(), stalled)
                try:
                    self._stream.abort()
                    self._stream.close()
                except Exception:
                    pass
                _safe(self.on_device_lost)
                try:
                    self._open()
                except Exception as e:
                    log.warning("reopen failed: %s", e)
                    next_try = now + 2.0
            self._stop.wait(0.005)


# ----------------------------------------------------------------------------- mic capture
class MicCapture:
    """Callback InputStream -> bounded queue -> consumer thread -> on_frame(frame_f32_mono, t_capture_monotonic).
    Default 48 kHz / 10 ms blocks so frames line up 1:1 with a 48 kHz player for AEC; the front-end
    downsamples to 16 kHz for VAD/STT."""

    def __init__(self, on_frame: Callable[[np.ndarray, float], None], device: "str | int | None" = None, *,
                 samplerate: int = 48000, blocksize: int = 480, latency: "float | str" = "low",
                 communications_mode: bool = False, backend=None, max_queue_s: float = 2.0):
        self._sd = backend or _real_sd()
        self.on_frame, self.device_query = on_frame, device
        self.samplerate, self.blocksize, self.latency = samplerate, blocksize, latency
        self.communications_mode = communications_mode
        self.stats = collections.Counter()
        self._q: "queue.Queue[tuple[float, np.ndarray]]" = queue.Queue(maxsize=int(max_queue_s * samplerate / blocksize))
        self._stream = None
        self._channels = 1
        self._closing = False
        self._lost = threading.Event()
        self._stop = threading.Event()
        self._last_cb = time.monotonic()
        self._thr: Optional[threading.Thread] = None

    def start(self) -> "MicCapture":
        self._open()
        self._thr = threading.Thread(target=self._consumer, name="mic-consumer", daemon=True)
        self._thr.start()
        return self

    def close(self) -> None:
        self._closing = True
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
        if self._thr:
            self._thr.join(timeout=1)

    def _open(self) -> None:
        sd = self._sd
        idx = resolve_device(self.device_query, "input", sd)
        extra = _extra_settings(sd, idx, self.communications_mode)
        last_err = None
        for ch in (1, 2):   # some endpoints refuse mono without auto-convert
            try:
                self._stream = sd.InputStream(
                    device=idx, samplerate=self.samplerate, channels=ch, dtype="float32",
                    blocksize=self.blocksize, latency=self.latency, extra_settings=extra,
                    callback=self._callback, finished_callback=self._on_finished)
                self._channels = ch
                break
            except Exception as e:
                last_err = e
        else:
            raise last_err  # type: ignore[misc]
        self._stream.start()
        self._last_cb = time.monotonic()
        self._lost.clear()
        log.info("input device %s opened (%d ch), latency %.1f ms", idx, self._channels, self._stream.latency * 1e3)

    def _on_finished(self) -> None:
        if not self._closing:
            self._lost.set()

    def _callback(self, indata, frames, time_info, status) -> None:
        now = time.monotonic()
        if status.input_overflow:
            self.stats["overflow"] += 1
        adc, cur = time_info.inputBufferAdcTime, time_info.currentTime
        t_cap = now - (cur - adc) if (adc > 0 and cur >= adc) else now - frames / self.samplerate
        x = indata[:, 0].copy() if self._channels == 1 else indata.mean(axis=1, dtype=np.float32)
        try:
            self._q.put_nowait((t_cap, x))
        except queue.Full:
            self.stats["dropped"] += 1
        self._last_cb = now

    def _consumer(self) -> None:
        next_try = 0.0
        while not self._stop.is_set():
            try:
                t, x = self._q.get(timeout=0.1)
                _safe(self.on_frame, x, t)
            except queue.Empty:
                pass
            now = time.monotonic()
            if (self._lost.is_set() or now - self._last_cb > 1.0) and now >= next_try:
                self.stats["device_lost"] += 1
                try:
                    self._stream.abort()
                    self._stream.close()
                except Exception:
                    pass
                try:
                    self._open()
                except Exception as e:
                    log.warning("mic reopen failed: %s", e)
                    next_try = now + 2.0


# ----------------------------------------------------------------------------- VAD + AEC
class SileroVAD:
    """Torch-free Silero VAD v5/v6 (ONNX graph inputs: input[1,64+512], state[2,1,128], sr int64).
    Model: https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"""
    WINDOW, CONTEXT, SR = 512, 64, 16000

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self._sr = np.array(self.SR, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._ctx = np.zeros((1, self.CONTEXT), np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self._ctx, np.asarray(window, np.float32).reshape(1, -1)], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._ctx = x[:, -self.CONTEXT:]
        return float(out[0, 0])


class EnergyVAD:
    """Dependency-free fallback: maps window dBFS to a pseudo-probability around threshold_dbfs."""

    def __init__(self, threshold_dbfs: float = -42.0):
        self.th = threshold_dbfs

    def reset(self) -> None:
        pass

    def __call__(self, window: np.ndarray) -> float:
        db = 10 * np.log10(float(np.mean(window * window)) + 1e-12)
        return float(np.clip((db - self.th) / 12.0 + 0.5, 0.0, 1.0))


class WebRtcAEC:
    """WebRTC AEC3 through livekit.rtc.AudioProcessingModule (pip install livekit; wheels for
    win_amd64 / manylinux / macOS, py3-none so any CPython>=3.9). Frames: exactly 10 ms int16."""

    def __init__(self, rate: int = 48000, ns: bool = True, hpf: bool = True, agc: bool = False, delay_hint_ms: int = 0):
        from livekit import rtc
        self._rtc, self.rate, self.n, self.delay = rtc, rate, rate // 100, delay_hint_ms
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=ns,
                                              high_pass_filter=hpf, auto_gain_control=agc)
        self._rev = np.zeros(0, np.float32)

    def _frame(self, f: np.ndarray):
        pcm = (np.clip(f, -1.0, 1.0) * 32767.0).astype(np.int16)
        return self._rtc.AudioFrame(pcm.tobytes(), self.rate, 1, self.n)

    def feed_reference(self, x: np.ndarray) -> None:
        self._rev = np.concatenate([self._rev, x.astype(np.float32, copy=False)])
        while self._rev.size >= self.n:
            f, self._rev = self._rev[:self.n], self._rev[self.n:]
            self._apm.process_reverse_stream(self._frame(f))

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(x.size, np.float32)
        for i in range(0, x.size - self.n + 1, self.n):
            fr = self._frame(x[i:i + self.n])
            self._apm.set_stream_delay_ms(self.delay)   # "must be called if echo processing is enabled"
            self._apm.process_stream(fr)
            out[i:i + self.n] = np.frombuffer(fr.data, np.int16) / 32768.0
        return out


# ----------------------------------------------------------------------------- front-end / barge-in
@dataclass
class FrontEndConfig:
    mode: str = "aec"            # "aec" | "energy_dtd" | "half_duplex"
    start_th: float = 0.5        # Silero prob to open a segment (idle)
    end_th: float = 0.35         # prob below which silence is counted
    barge_th: float = 0.7        # stricter prob while the AI is speaking
    min_speech_ms: int = 160     # consecutive speech needed to emit speech_start
    barge_min_ms: int = 250      # sustained speech while AI talks before emitting barge_in
    min_silence_ms: int = 600    # trailing silence that closes an utterance (end-of-turn)
    preroll_ms: int = 300
    max_utt_s: float = 20.0
    dtd_k: float = 2.0           # energy_dtd: mic_rms must exceed k * coupling * ref_rms
    dtd_window_s: float = 0.35   # look-back over reference to cover output+acoustic+input delay


class VoiceFrontEnd:
    """mic 10 ms frames @in_rate -> [AEC3] -> 16 kHz -> 32 ms windows -> VAD -> events.
    on_event(kind, t_monotonic, payload): "speech_start" {barge:bool} | "barge_in" {} | "speech_end" np.ndarray(16k f32)
    Feed from MicCapture(on_frame=frontend.feed)."""

    def __init__(self, vad: Callable[[np.ndarray], float], on_event: Callable[[str, float, Any], None],
                 player: Optional[StreamingPlayer] = None, in_rate: int = 48000, aec: Optional[WebRtcAEC] = None,
                 cfg: FrontEndConfig = FrontEndConfig()):
        self.vad, self.on_event, self.player, self.in_rate, self.aec, self.cfg = vad, on_event, player, in_rate, aec, cfg
        self._rs = None
        if in_rate != 16000:
            import soxr
            self._rs = soxr.ResampleStream(in_rate, 16000, 1, dtype="float32", quality="HQ")
        self._buf16 = np.zeros(0, np.float32)
        self._win_ms = 32
        self._pre: collections.deque = collections.deque(maxlen=max(1, cfg.preroll_ms // self._win_ms))
        self._utt: list = []
        self._in_speech = False
        self._speech_run = 0
        self._sil_run = 0
        self._barge_sent = False
        self._coupling = 0.0
        self._ref_seen = 0.0     # monotonic time of last reference block pushed into AEC

    def feed(self, frame: np.ndarray, t_cap: float) -> None:
        if self.aec is not None and self.player is not None:
            ref = self.player.reference
            while ref and ref[0][0] <= self._ref_seen:     # drop blocks already consumed
                ref.popleft()
            for t, blk in list(ref):                       # render must be fed before its echo is captured
                self.aec.feed_reference(blk)
                self._ref_seen = t
            frame = self.aec.process(frame)
        x = self._rs.resample_chunk(frame) if self._rs is not None else frame
        self._buf16 = np.concatenate([self._buf16, x])
        while self._buf16.size >= 512:
            w, self._buf16 = self._buf16[:512], self._buf16[512:]
            self._window(w, t_cap)

    def _window(self, w: np.ndarray, t: float) -> None:
        c = self.cfg
        p = self.vad(w)
        speaking = self.player.is_speaking() if self.player is not None else False
        if speaking and c.mode == "half_duplex":
            p = 0.0
        elif speaking and c.mode == "energy_dtd" and self.player is not None:
            mic_rms = float(np.sqrt(np.mean(w * w)))
            ref_rms = self.player.ref_rms_max(t - c.dtd_window_s, t)
            if ref_rms > 1e-3:
                if p < 0.2:   # AI-only period: learn acoustic coupling (peak-hold, slow decay)
                    self._coupling = max(self._coupling * 0.995, mic_rms / ref_rms)
                if mic_rms < c.dtd_k * self._coupling * ref_rms:
                    p = 0.0
        th_on = c.barge_th if speaking else c.start_th
        is_sp = p >= (th_on if not self._in_speech else c.end_th)
        if not self._in_speech:
            self._pre.append(w)
            self._speech_run = self._speech_run + 1 if is_sp else 0
            if self._speech_run * self._win_ms >= c.min_speech_ms:
                self._in_speech, self._sil_run, self._barge_sent = True, 0, False
                self._utt = list(self._pre)
                self.on_event("speech_start", t, {"barge": speaking})
        else:
            self._utt.append(w)
            self._speech_run = self._speech_run + 1 if is_sp else 0
            if speaking and not self._barge_sent and self._speech_run * self._win_ms >= c.barge_min_ms:
                self._barge_sent = True
                self.on_event("barge_in", t, {})
            self._sil_run = 0 if is_sp else self._sil_run + 1
            too_long = len(self._utt) * self._win_ms >= c.max_utt_s * 1000
            if self._sil_run * self._win_ms >= c.min_silence_ms or too_long:
                audio = np.concatenate(self._utt)
                self._in_speech, self._utt, self._speech_run = False, [], 0
                self._pre.clear()
                self.on_event("speech_end", t, audio)

WIRING on the Windows PC (see audio_demo_windows.py):
  player = aio.StreamingPlayer("Speakers", on_level=lambda r: loop.call_soon_threadsafe(vts_mouth, r)).start()
  aec    = aio.WebRtcAEC(48000)   # ImportError -> cfg.mode='energy_dtd', aec=None
  fe     = aio.VoiceFrontEnd(aio.SileroVAD(r"models\silero_vad.onnx"), on_event, player=player, aec=aec, cfg=aio.FrontEndConfig(mode="aec"))
  mic    = aio.MicCapture(fe.feed, "Microphone").start()
  TTS adapter: for each sentence, player.play(pcm_f32_mono, tts_sr) per chunk, then player.mark(lambda ok, s=sentence: ...). Interrupt: player.cancel().
  speech_end payload = 16 kHz float32 utterance for the STT component (includes ~300 ms pre-roll).

## Latency & resources

Playback path, from queued PCM to sound:
- Callback block: 10 ms (480 frames at 48 kHz).
- WASAPI shared engine period: 10 ms. PortAudio enforces period >= DefaultDevicePeriod.
- Requested latency=0.04 gives stream.latency around 40 ms. On ALSA null it reported exactly 0.040. On Windows WASAPI I estimate 20-50 ms; read stream.latency on the PC.
- cancel() to silence: at most 1 block (10 ms) plus the device buffer, so about 30-60 ms (estimate). In the fake-backend test the block right after cancel was fade + zeros.
- For comparison, MME default is about 90 ms and DirectSound about 120 ms (PortAudio defaults). VB-CABLE's internal buffer defaults to 7168 samples, up to ~149 ms at 48 kHz. That can shift the OBS copy against the speaker copy; fix with the OBS Sync Offset or the VB control panel Latency.

Capture path:
- 10 ms blocks; WASAPI low input latency (stream.latency about 10 ms or more).
- Queue hop to the consumer thread: under 1 ms.
- soxr HQ 48k to 16k: ~4 µs CPU, but output arrives in bursts, up to ~30-40 ms added delay.
- VAD window: 32 ms.
- Decisions: speech_start needs 160 ms of speech; barge_in needs 250 ms sustained; end-of-turn needs 600 ms of silence (tunable).
- Measured barge-in reaction (synthetic): about 0.3 s to speech_start and 0.4 s to barge_in in AEC mode; about 0.5 s and 0.6 s in energy_dtd mode.

CPU cost per 10 ms (container CPU; the i7-14700KF P-cores should be faster):
- livekit AEC3: 90 µs at 16 kHz, 137 µs at 48 kHz (forward + reverse). About 1.4% of one core at 48 kHz.
- pyaec speex: 100-120 µs at 16 kHz, due to list conversion.
- Silero ONNX: 0.15 ms per 32 ms window, about 0.5% of one core.
- samplerate sinc_fastest: 37 µs.
- Python callbacks: negligible.
- Total audio front-end: under 3% of one core. No GPU used; the whole path is CPU-only and leaves the RTX 4070 (12 GB) for the LLM, STT and TTS.

Memory:
- livekit wheel 10.7 MB; it imports in about 0.18 s and starts a tokio FFI runtime.
- onnxruntime is ~tens of MB RSS; the Silero model is 2.3 MB.
- Player reference ring: 3 s × 48 kHz float32 ≈ 0.6 MB.

Disk: sounddevice wheel 985 KB (bundles the PortAudio DLLs), soxr 170 KB, pyaec 80 KB, aec-audio-processing 899 KB.

Threads per player/mic pair:
- 2 PortAudio threads (MMCSS priority in WASAPI)
- 1 player notifier/watchdog (5 ms poll)
- 1 mic consumer that runs AEC + VAD
- the livekit FFI runtime threads.

## Pitfalls

- sd.default / device=None / plain name strings resolve to MME (PortAudio's first Windows host API). That means 90 ms+ latency, 31-char truncated names, and 'Multiple devices found' ValueError because each device is listed 3-4 times (MME/DS/WASAPI/WDM-KS). Resolve to a WASAPI index yourself.
- WASAPI shared mode without WasapiSettings(auto_convert=True) fails to open when samplerate/channels differ from the endpoint mix format (e.g. mic set to 44.1 kHz). Passing WasapiSettings to a non-WASAPI device raises. exclusive=True locks the device away from OBS and Discord.
- If the callback raises, PortAudio never calls it again (silent death). Wrap the body in try/except and always fill outdata. Never call stream.close/abort/stop from finished_callback or the audio callback.
- The PortAudio device list is frozen at init. Hot-plugged or renumbered devices ('2- USB Audio Device') and Windows default-device changes are invisible until sd._terminate() + DLL reload + sd._initialize(). This is a private API and kills ALL streams, so one supervisor must close and reopen both player and mic. Select devices by name substring, never by stored index.
- Python GIL: heavy pure-Python work in the same process (tokenizers, JSON floods, big numpy copies holding the GIL) delays the audio callback, causing underflow clicks. Keep latency at or above 0.04 and watch player.stats['underflow']. Move audio into a separate process if underflows persist. onnxruntime, ctranslate2 and llama.cpp bindings release the GIL during compute.
- soxr.ResampleStream holds output back and emits in bursts (~30 ms at HQ with 10 ms chunks). The tail is lost unless resample_chunk(empty, last=True) is called; StreamingPlayer.mark() does this. Recreate the stream after cancel().
- The silero-vad pip package hard-depends on torch (a multi-GB install). Use the ONNX file with onnxruntime (wrapper in audio_io.SileroVAD). The v5/v6 graph needs a 64-sample left context prepended; exactly 512 samples at 16 kHz.
- AEC only removes what is in its reference. The reference here is the AI voice only. Game audio, music, Discord and alerts from the speakers still reach the mic, and Silero fires on game or Discord voices. Options: half_duplex mode, push-to-talk, headphones, or a WASAPI-loopback full-mix reference via PyAudioWPatch fed to process_reverse_stream (not implemented or tested).
- The AEC reference must be the samples that reach the speakers, after player.gain (the tap is taken after gain). When a second sink (VB-CABLE) is used, feed only the speaker player's reference. Loud or clipping speakers, nonlinear laptop speakers and mic auto-gain reduce ERLE; my benchmark was linear.
- Mic and speakers on different clocks (USB mic vs motherboard codec) drift over hours. AEC3 re-estimates delay, but multi-hour behaviour is untested; run a 2 h soak test.
- livekit APM: frames must be exactly 10 ms int16 at 8, 16, 32 or 48 kHz. process_stream modifies the frame in place. Call set_stream_delay_ms before each process_stream when AEC is on. On interpreter exit livekit prints a harmless AssertionError from FfiHandle.__del__, and soxr prints nanobind 'leaked instance' warnings; both are cosmetic.
- pyaec (speex) needs the echo delay to fit inside filter_length (tail). With 4096 taps at 16 kHz (256 ms) it failed completely at 300 ms bulk delay, and its preprocessor suppresses streamer speech during double-talk. Fallback only.
- aec-audio-processing has Windows wheels for cp311-313 only and no Linux wheels. Ubuntu CI would need swig + meson to build from the sdist. It is also unmaintained-looking (no project URL).
- VB-CABLE install can silently become the Windows default playback AND recording device. Everything then goes into the cable: the streamer hears nothing, and the 'default mic' becomes CABLE Output, so the AI would hear itself. Always select devices explicitly by name. Default cable buffer is 7168 samples (~149 ms max).
- OBS doubling: if the AI plays on the default speakers (captured by Desktop Audio) and is also captured through VB-CABLE or Application Audio Capture, the voice is doubled or phased. Use one capture path per track. Application Audio Capture picks a window and captures that window-owner's process tree. A console python.exe has no own window; whether picking the terminal window captures it is untested.
- Tagging capture as AudioCategory_Communications (experimental flag) triggers Windows default ducking (other sounds -80%, e.g. game and music on stream). It may apply Voice Clarity or OEM AEC with an uncontrollable reference endpoint. Stacking it with livekit AEC3 can double-process. Default is off.
- Bluetooth headsets: opening the headset mic switches it to the HFP profile (narrowband, degraded playback). Prefer a wired or USB mic.
- Timestamps: outputBufferDacTime and inputBufferAdcTime may be 0 on MME/DS; the code falls back to stream.latency. Lip-sync and markers are therefore only DAC-accurate on WASAPI.
- CI: `import sounddevice` raises OSError on ubuntu runners without libportaudio2, so keep imports lazy and inject FakeSD. Whether windows-latest runners expose any audio endpoint is unverified; never open real streams in CI.
- Barge-in policy is an architecture decision, not DSP. Instant cancel on any speech makes the AI stop on coughs or laughs. Recommended: duck at speech_start, cancel at sustained barge_in (250 ms or more), optionally only if the STT text is addressed to her. Record which sentence markers fired True so the LLM history is cut to what was heard.

## Open questions

- What are the real-room numbers on the user's PC (ERLE, false barge-in rate, detection latency) with their actual speakers, mic and distance? Run audio_demo_windows.py in modes aec, energy_dtd and half_duplex and log player.stats and mic.stats.
- What stream.latency and underflow rate does WASAPI shared mode give on the user's Realtek or USB devices at latency=0.04 and blocksize=480? Would 0.03 still be glitch-free while the LLM, STT and TTS run in the same process?
- Does the streamer play game audio or music through the same speakers? If so, a WASAPI-loopback full-mix reference (PyAudioWPatch) or half_duplex while gaming becomes necessary.
- Does tagging the PortAudio capture stream AudioCategory_Communications on Windows 11 24H2/25H2 actually enable Voice Clarity or OEM AEC? Which render endpoint does it use as reference, and does the -80% ducking hit OBS captures?
- Does OBS Application Audio Capture catch python.exe launched from Windows Terminal or a conhost window (process-tree semantics), or does the app need its own window, for example a native control panel?
- Does sd._terminate() + sd._initialize() alone refresh the device list on Windows 19.7.0 (issue #516 says no), or is the DLL reload always needed? And does it disturb other open streams, such as ones owned by livekit? (It doesn't open audio devices, so probably no.)
- Does AEC3 hold up over a multi-hour stream when the USB mic and speaker codec clocks drift?
- Do GitHub Actions windows-latest runners expose any WASAPI endpoint? This matters only if someone adds hardware-touching tests; it's irrelevant with FakeSD.
- Architecture: should barge-in cancel on voice at all (Neuro's documented interruption is priority-based from the operator/system), or only duck and let the LLM decide after the STT text arrives?
