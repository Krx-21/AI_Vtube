# Live2D avatar control — component brief

_Research snapshot: 2026-09-25. Verified facts carry a source; unverified items are marked._

## Recommendation

DEFAULT: use VTube Studio (Steam app 1325860, latest announced 1.35.10 on 2026-07-08) as a separate renderer process that can be restarted on its own. Pailin's Python "brain" drives it through the VTS Public API on ws://127.0.0.1:8001 with our own client of about 150 lines, built on `websockets` 17.1 (asyncio). Do not use pyvts. This matches what Neuro's rewrite did: after it, the character logic survives a renderer restart (T1, Mar/Apr 2026). The brain holds all state, and the renderer is a sink that can be swapped out.

How it works:
(1) LIP-SYNC. We own the TTS PCM. Before a speech segment plays, compute two tracks at 60 Hz: MouthOpen (RMS → dBFS → normalise between -45 and -12 dB → gamma 0.7 → one-pole attack 30 ms / release 80 ms → shifted 40 ms earlier to cover render and capture latency) and a heuristic MouthForm (high-band to low-band energy ratio, for i/e versus o/u). A 60 Hz AvatarDriver task reads the playback position from the audio player and sends InjectParameterDataRequest with the default input IDs MouthOpen and MouthSmile. It uses faceFound=true and mode "set", and pipelines messages without awaiting each response. VTS's model auto-setup already maps MouthOpen→ParamMouthOpen and MouthSmile→ParamMouthForm, so any model works with no setup. When speech is interrupted, the player stops, the envelope lookup returns 0, and the mouth closes within the release time.
(2) IDLE / "ALIVE" MOTION. There is no face tracker. The driver also injects procedural FaceAngleX/Y/Z, FacePositionX and EyeLeft/Right X/Y (slow sines, random saccades, small nods scaled by speech level). It injects EyeOpenLeft/Right at about 1.0 with VTS auto-blink turned on for the eye outputs, and leaves ParamBreath on VTS auto-breath and body sway on the model's idle .motion3.json. The corpus notes Neuro's body motion is a pre-recorded loop (T5).
(3) EMOTION. The LLM emits one tag from a small set per sentence (e.g. [happy] [sad] [angry] [surprised] [shy] [smug] [neutral]). The tag is stripped before TTS and applied when that sentence's audio starts. Emotion is applied two ways: (a) a baseline added to the injected MouthSmile and Brows, which works on any model; (b) an idempotent expression state machine using ExpressionActivationRequest with explicit activate/deactivate and fadeTime 0.3. Do not use ToggleExpression hotkeys for this: firing one twice turns the expression off. Optional one-shot reaction animations go through HotkeyTriggerRequest by hotkey name; errors from cooldown or a full queue are ignored.
(4) TOOLS like Neuro's "spin model": MoveModelRequest with relative rotations. Operator panel: HotkeyTriggerRequest and ExpressionStateRequest.
(5) OBS: capture via Spout2 (the VTS-recommended method on Windows) with Premultiplied Alpha.
(6) SECOND CHARACTER (evil twin): run a second VTS instance. It auto-binds the next free port; find it with the UDP 47779 discovery broadcast.

Why VTS by default:
- No rendering code to write.
- A well-documented, stable API (still apiVersion "1.0").
- Built-in physics, idle animation, auto-blink, auto-breath and expressions.
- It is officially licensed to use the Live2D Cubism SDK, so the Live2D licensing burden sits with DenchiSoft. The streamer only buys the $14.99 "Remove Watermark" DLC if streams are monetized: the FAQ says commercial use requires a paid version. The watermark itself only appears while webcam tracking is active.
- A browser/Cubism renderer that we ship ourselves is, by Live2D's own definition, an "Expandable Application" (avatars / live-streaming apps). The individual / small-business exemption does not apply to those, and Core cannot be redistributed under an open license.

Keep an AvatarSink interface with three implementations:
- VTSSink (default).
- BrowserSink: a Python WebSocket feeding an OBS Browser Source page built on pixi.js 7.4.3 + pixi-live2d-display-lipsyncpatch 0.5.0-ls-8. Cubism Core is downloaded by the user and never committed. This is the fallback when VTS is unavailable or unwanted, same shape as Open-LLM-VTuber's renderer.
- FakeSink / FakeVTS server for CI on ubuntu and windows.

Test models: Live2D samples Mao (8 expressions exp_01..08, LipSync param ParamA) and Haru (8 expressions F01..F08) for emotion mapping; Hiyori (no expressions, 9 Idle motions) for plain lip-sync. All three are Live2D Original Characters: commercial use is allowed for individuals and small businesses (<10M JPY sales) if the copyright notice is shown. VTS's bundled Akari is for test streams only. Pailin needs a commissioned model with standard parameter IDs; A/I/U/E/O vowel blendshape parameters are nice to have.

## Alternatives

### VTS + own minimal websockets client + injected MouthOpen/MouthSmile/FaceAngle from our TTS PCM (RECOMMENDED)
- **Pros:** No rendering code. Physics, idle animation, auto-blink, auto-breath and expressions are built in. Live2D licensing is carried by a licensed app. Renderer restarts separately from the brain (Neuro-like). Deterministic, pre-computed lip-sync with a lead can be aligned to our own interruptible player. Test models are free.
- **Cons:** Windows/macOS only, a closed-source Steam app. $14.99 DLC for monetized use. Mouth shapes are limited to open plus form unless the model has vowel parameters and they can be injected (unverified). Hotkey queue and cooldown quirks.
- **When:** Default for the streamer's Windows 11 PC.

### VTS Advanced Lipsync via virtual audio cable (e.g. VB-CABLE): duplicate TTS output to the cable and select it as the VTS microphone
- **Pros:** Real A/I/U/E/O vowel shapes from uLipSync with no lip-sync code. Works for singing and any audio.
- **Cons:** Extra driver install and routing. Device buffering latency, and the mouth cannot lead the audio. Needs calibration to the TTS voice. VTS does not document Thai vowels. A second audio path to keep in sync and to stop on interruption.
- **When:** Optional 'high quality mouth' mode once the model has ParamA/I/U/E/O blendshapes, or for karaoke segments.

### Browser renderer: pixi.js 7.4.3 + pixi-live2d-display-lipsyncpatch 0.5.0-ls-8 page as an OBS Browser Source, fed by a Python WebSocket (the Open-LLM-VTuber approach)
- **Pros:** Fully under our control, no Steam app, no watermark. Parameters, expressions and motions are driven directly. Restartable independently. Cubism 3-5 models.
- **Cons:** Cubism Core is proprietary: users must download it, and publishing an arbitrary-model streaming app is an 'Expandable Application' needing Live2D approval. The library forks are sparsely maintained and locked to pixi v6/v7. We must build idle, blink and breath behaviour ourselves. Core-version compatibility is unverified.
- **When:** Fallback when VTS is unavailable, or for a fully self-hosted demo; keep it behind the AvatarSink interface.

### easy-live2d 1.0.0 (pixi 8 + Cubism 5 SDK for Web R5 Core)
- **Pros:** Modern pixi 8 and the current Cubism R5 framework. Has lip-sync, expressions and motions. MIT.
- **Cons:** Released 2026-09-24, one day before this research. Same Core licensing issue. Needs WebGL2. Core must match exactly.
- **When:** Revisit in a few months as the browser-renderer base if pixi v7 lock-in becomes a problem.

### live2d-py 0.8.0.9 native renderer (OpenGL in Python, captured by OBS game capture or Spout)
- **Pros:** All Python. Windows abi3 wheel. Cubism 2.1 and 3+. Direct parameter control.
- **Cons:** Rendering in the brain process breaks the decoupling unless it runs as a separate process. Same Core licensing issue. Needs its own capture and transparency plumbing.
- **When:** Only if neither VTS nor a browser is acceptable.

### pyvts 0.3.3 / coovts 0.1.0 libraries instead of our own client
- **Pros:** Ready-made request builders. coovts is typed with pydantic.
- **Cons:** pyvts: naive send/recv without request matching, no event demultiplexing, opencv dependency, last release Sep 2024. coovts: brand new, Python>=3.12, extra pydantic dependency.
- **When:** Not recommended; our client is about 150 lines and covered by tests.

## Verified facts

- ✅ VTS API websocket server default is ws://localhost:8001 and the port is user-changeable. VTS always replies with text frames. The user must enable 'Allow Plugin API access' / Start API in VTS. Every request carries apiName 'VTubeStudioPublicAPI' and apiVersion '1.0'. The version stays '1.0' until an incompatible change, and new fields can appear without a bump, so deserialization must ignore unknown fields.  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/README.md (API Details)
- ✅ requestID is optional. If given it must be 1-64 ASCII chars; if omitted, VTS generates a UUID. Every response has 'timestamp' (ms) and a 'data' object (empty {} when there is no data).  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/README.md
- ✅ Auth flow: AuthenticationTokenRequest{pluginName, pluginDeveloper (each 3-32 chars), optional pluginIcon = base64 PNG/JPG exactly 128x128}. This shows an Allow/Deny popup in VTS. Allow returns AuthenticationTokenResponse{authenticationToken (ASCII, <=64 chars)}. Deny returns APIError errorID 50. Then each session sends AuthenticationRequest{pluginName, pluginDeveloper, authenticationToken}, which returns {authenticated: true|false, reason}. Name and developer must match the values used for the token. Reuse the token across sessions. If the user revokes it, authenticated=false and you re-request.  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/README.md (Authentication)
- ✅ InjectParameterDataRequest{faceFound, mode:'set'|'add', parameterValues:[{id, value, weight?}]}. Values must be in [-1e6, 1e6]. weight is 0..1 and only used in set mode (default 1). The API overrides tracking values while data keeps arriving. Each parameter must be re-sent 'at least once every second', otherwise it counts as lost and reverts to its previous controller or its default. faceFound:true makes VTS treat the face as found, which controls the tracking-lost animation. Only one plugin may 'set' a given parameter (error 454 InjectDataParamControlledByOtherPlugin). Any number of plugins may use 'add'. An unknown mode gives error 455.  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/README.md (Feeding in data...) + Files/ErrorID.cs
- ✅ Default input parameters: FacePositionX/Y/Z, FaceAngleX/Y/Z, MouthSmile, MouthOpen, Brows, MousePositionX/Y, TongueOut, EyeOpenLeft/Right, EyeLeftX/Y, EyeRightX/Y, CheekPuff, BrowLeftY, BrowRightY, VoiceFrequency, VoiceVolume, VoiceVolumePlusMouthOpen, VoiceFrequencyPlusMouthSmile, VoiceA, VoiceI, VoiceU, VoiceE, VoiceO, VoiceSilence, MouthX, FaceAngry. The MouthOpen input range is [0,1]. Custom parameters: unique, alphanumeric, 4-32 chars; at most 100 per plugin and 300 globally; default names are not allowed (error 354).  
  Source: https://raw.githubusercontent.com/wiki/DenchiSoft/VTubeStudio/VTS-Model-Settings.md + README (ParameterCreationRequest)
- ✅ VTS 'Advanced Lipsync' is based on hecomi/uLipSync, works on all platforms and is calibrated per voice. It outputs VoiceA/I/U/E/O/Silence/Volume/Frequency in 0..1. 'Simple Lipsync' (Oculus OVRLipSync, Windows-only) is legacy and not recommended.  
  Source: https://raw.githubusercontent.com/wiki/DenchiSoft/VTubeStudio/Lipsync.md
- ✅ Parameter value priority in VTS, low to high: P0 default, P1 idle animation, P2 face tracking, P3 one-time animation, P4 expression, P5 physics. Control hand-offs fade smoothly. Expression parameters support Overwrite/Add/Multiply modes; all multiplies are applied first, then all adds.  
  Source: https://raw.githubusercontent.com/wiki/DenchiSoft/VTubeStudio/Interaction-between-Animations,-Tracking,-Physics,-etc..md
- ✅ HotkeyTriggerRequest{hotkeyID: ID or case-insensitive name, itemInstanceID?} returns HotkeyTriggerResponse{hotkeyID}. Per the README, one specific hotkey can only fire once every 5 frames, and the queue executes one hotkey every 5 frames with a capacity of 32 (error 200 when full). CONFLICT: the ErrorID.cs comment says HotkeyCooldownNotOver(203) is a 'global 5 second cooldown'. HotkeysInCurrentModelRequest{modelID?, live2DItemFileName?} returns availableHotkeys[{name,type,description,file,hotkeyID,keyCombination (always empty),onScreenButtonID}].  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/README.md + Files/ErrorID.cs
- ✅ Hotkey action types: Unset, TriggerAnimation, ChangeIdleAnimation, ToggleExpression, RemoveAllExpressions, MoveModel, ChangeBackground, ReloadMicrophone, ReloadTextures, CalibrateCam, ChangeVTSModel, TakeScreenshot, ScreenColorOverlay, RemoveAllItems, ToggleItemScene, DownloadRandomWorkshopItem, ExecuteItemAction, ArtMeshColorPreset, ToggleTracker, ToggleTwitchFeature, LoadEffectPreset, ToggleLive2DEditorAPI, WebItemAction, ToggleModelSound.  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/Files/HotkeyAction.cs
- ✅ ExpressionStateRequest{details, expressionFile?} returns expressions[{name,file,active,deactivateWhenKeyIsLetGo,autoDeactivateAfterSeconds,secondsRemaining,usedInHotkeys,parameters}]. ExpressionActivationRequest{expressionFile:'x.exp3.json', fadeTime (clamped 0-2, default 0.25), active} returns an empty response; errors 650/651/652. The fade-out reuses the fade-in time. The docs recommend hotkeys over direct activation so users can always turn expressions off.  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/README.md
- ✅ CurrentModelRequest returns {modelLoaded, modelName, modelID, vtsModelName, live2DModelName, numberOfLive2DParameters, ..., modelPosition{positionX,positionY,rotation,size}}. MoveModelRequest{timeInSeconds 0-2, valuesAreRelativeToModel, positionX/Y (-1000..1000), rotation (-360..360), size (-100..100)}. It can be sent every frame with timeInSeconds 0.  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/README.md
- ✅ Event API: EventSubscriptionRequest{eventName, subscribe, config}. Events: TestEvent, ModelLoadedEvent, TrackingStatusChangedEvent, BackgroundChangedEvent, ModelConfigChangedEvent, ModelMovedEvent, ModelOutlineEvent (fixed 15 FPS), HotkeyTriggeredEvent, ExpressionToggledEvent, ModelAnimationEvent, ItemEvent, ModelClickedEvent, PostProcessingEvent, Live2DCubismEditorConnectedEvent, ArtMeshTrackingEvent (1-60 Hz), ArtMeshOutlineEvent (1-30 Hz). ExpressionToggledEvent, ArtMeshTrackingEvent and ArtMeshOutlineEvent are PUBLIC-BETA ONLY. Subscriptions end on disconnect, and event-config errors are offset by 100000. The example event payloads carry a requestID, so a client must not match events to pending requests by requestID alone.  
  Source: https://raw.githubusercontent.com/DenchiSoft/VTubeStudio/master/Events/README.md
- ✅ No websocket message-size or request-rate limit is documented (grep of README and Events). Documented soft limits: InputParameterListRequest should not be sent at 60+ FPS; PostProcessingUpdateRequest not every frame; custom item image data under 5 MB. Client-side websockets 17.1 defaults: max_size 1 MiB, ping_interval 20 s, ping_timeout 20 s, compression deflate, proxy=True (introspected in the container).  
  Source: README.md grep + websockets 17.1 inspect.signature
- ✅ UDP discovery: VTS broadcasts VTubeStudioAPIStateBroadcast{active, port, instanceID, windowTitle} on UDP 47779 every 2 s, even when the API is off. Extra VTS instances bind the next free port and get titles 'VTube Studio Window 2', and so on. They can be started via start_without_steam.bat or the 'Start VTube Studio' button.  
  Source: README.md (API Server Discovery) + https://raw.githubusercontent.com/wiki/DenchiSoft/VTubeStudio/Starting-without-Steam.md
- ✅ Latest announced VTS Steam release is 1.35.10 (2026-07-08). Earlier: 1.35.7 (2026-05-20), 1.35.0 SFX system (2026-03-31), 1.34.0 ArtMesh groups (2026-03-28). Since 1.32.57 (2025-10-04) VTS uses Unity 6000.0.58f2 (fix for CVE-2025-59489). A later unannounced patch cannot be excluded.  
  Source: https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=1325860
- ✅ VTS on Steam is free with all features; Windows and macOS only, no Linux; minimum spec DX11 GPU with 2 GB VRAM. DLC 'Remove Watermark' is $14.99 and 'VNet Multiplayer Collab' is $19.99. The watermark shows only while webcam tracking is active. FAQ: monetized use (superchats, monetized Twitch) requires buying at least one paid version 'due to Live2D licensing'. Bigger companies need extra licensing per the EULA.  
  Source: https://store.steampowered.com/api/appdetails?appids=1325860 ; https://raw.githubusercontent.com/wiki/DenchiSoft/VTubeStudio/FAQ.md
- ✅ Bundled VTS models: Akari (VTS mascot, by Denchi) may be used in test streams only; no commercial use and no building a VTuber identity on her. All other bundled models are (c) Live2D Inc. under Live2D sample terms. Hiyori being bundled is confirmed by Denchi in a Steam thread. The full bundled list is NOT verified.  
  Source: https://raw.githubusercontent.com/wiki/DenchiSoft/VTubeStudio/Privacy-Policy,-Licensing-and-Further-Terms.md ; https://steamcommunity.com/app/1325860/discussions/0/3457094584926683767
- ✅ OBS capture: on Windows, Spout2 is the recommended method: fast, transparent, excludes the VTS UI. Needs Off-World-Live obs-spout2-plugin matching the OBS version. Set Composite mode to 'Premultiplied Alpha', and set the VTS background to transparent black / Color Picker with 'Transparent in capture'. Alternatives: Game Capture with transparency, NDI, or Virtual Webcam.  
  Source: https://raw.githubusercontent.com/wiki/DenchiSoft/VTubeStudio/Recording-Streaming-with-OBS.md
- ✅ pyvts 0.3.3 (PyPI, uploaded 2024-09-10, MIT) requires websockets>=10.4, aiofiles>=23.1.0 and opencv-python>=4.4.0 (OpenCV used only to resize the icon). Its vts.request() does send() then recv() with no requestID matching or event demultiplexing, and event_subscribe() just calls request(). Last release was about 2 years ago.  
  Source: https://pypi.org/pypi/pyvts/json ; https://raw.githubusercontent.com/Genteki/pyvts/main/setup.py ; pyvts/vts.py
- ✅ coovts 0.1.0 (PyPI, 2026-09-19) requires Python>=3.12, pydantic>=2.13.5 and websockets>=17.1. The VTS README lists it as 'in early dev'.  
  Source: https://pypi.org/pypi/coovts/json ; VTS README plugin table
- ✅ websockets 17.1 was released 2026-08-26 and 17.0 on 2026-07-29. 17.0 requires Python>=3.11 (16.1 was the last for 3.10). 15.0 made clients use SOCKS/HTTP proxies automatically. 14.0 made the new asyncio implementation the default and deprecated the legacy one.  
  Source: https://pypi.org/pypi/websockets/json ; https://websockets.readthedocs.io/en/stable/project/changelog.html
- ✅ Cubism Core is under the Live2D Proprietary Software License (v2.1, revised 2025-02-03). 'Expandable Application' covers derivative works that load an indefinite number of models, e.g. avatars and live-streaming apps. Publishing one needs prior application, approval and a Publication License, and the General User / Small-Scale Enterprise (<10M JPY sales) exemption does not apply. Clause 5.3.2 forbids redistributing the Redistributable Code in a way that places it under an 'excluded license' (one requiring source disclosure or allowing third-party modification).  
  Source: https://www.live2d.com/eula/live2d-proprietary-software-license-agreement_en.html
- ✅ Cubism Web Framework 5-r.5 was released 2026-04-02 (compatible with Cubism 5.3). Cubism Core for Web is NOT in the GitHub repos and must be downloaded from live2d.com. Core was upgraded to 06.00.0001 on 2026-01-08. Framework and Samples are under the Live2D Open Software License. Businesses with at least 10M JPY annual revenue need the Cubism SDK Release License.  
  Source: https://raw.githubusercontent.com/Live2D/CubismWebFramework/develop/CHANGELOG.md, LICENSE.md ; https://raw.githubusercontent.com/Live2D/CubismWebSamples/develop/README.md, Core/CHANGELOG.md, Core/LICENSE.md
- ✅ pixi-live2d-display: latest 0.4.0 (2022-09-04) and beta 0.5.0-beta (2023-12-07), with pixi v6 peers. Fork pixi-live2d-display-lipsyncpatch 0.5.0-ls-8 (2025-06-02, MIT) has peer pixi.js ^7 and adds model.speak(url,{volume,expression,resetExpression,crossOrigin,onFinish,onError}) and model.stopSpeaking(). Its UMD build registers PIXI.live2d. InternalModel emits 'beforeModelUpdate' after motions/physics and before model.update(), which is where coreModel.setParameterValueById overrides go. The README warns the cubism.live2d.com Core hotlink is sometimes down and should not be used in production. pixi.js latest is 8.21.0; the 7.x line ends at 7.4.3.  
  Source: https://registry.npmjs.org/pixi-live2d-display , /pixi-live2d-display-lipsyncpatch (tarball dist/cubism4.es.js inspected), /pixi.js
- ✅ Open-LLM-VTuber (v1.2.1; a v2 rewrite is in planning) renders with pixi-live2d-display-lipsyncpatch and supports Cubism 3-5 only, not Cubism 2. model_dict.json has name, url, kScale, idleMotionGroupName, emotionMap {emotion: expression index/name}, tapMotions and defaultEmotion. The LLM writes [emotion] tags, which are mapped to expressions and stripped. For lip-sync the backend sends WAV plus 'volumes' = pydub RMS per 20 ms chunk normalised to the chunk max ('slice_length': 20). It bundles mao_pro with a Live2D sample-data notice.  
  Source: https://raw.githubusercontent.com/Open-LLM-VTuber/Open-LLM-VTuber/main/{README.md,model_dict.json,src/open_llm_vtuber/utils/stream_audio.py,src/open_llm_vtuber/live2d_model.py} ; https://docs.llmvtuber.com/docs/user-guide/live2d
- ✅ moeru-ai/airi packages/stage-ui-live2d depends on pixi-live2d-display plus modular @pixi/* v6-style packages.  
  Source: https://raw.githubusercontent.com/moeru-ai/airi/main/packages/stage-ui-live2d/package.json
- ✅ easy-live2d 1.0.0 (npm, 2026-09-24, MIT) needs Pixi.js 8 and WebGL2 and requires the matching Cubism 5 SDK for Web R5 Core. Pin 0.4.4 for older Cores.  
  Source: https://registry.npmjs.org/easy-live2d
- ✅ live2d-py 0.8.0.9 (PyPI, 2026-09-24) needs Python>=3.11 and ships a cp311-abi3 win_amd64 wheel. It is a native (non-web) Live2D renderer for any OpenGL context (pygame/PySide6/GLFW), supporting Cubism 2.1 and 3+, with lip-sync and parameter control. Its repo excludes Cubism Core and Framework for license reasons.  
  Source: https://pypi.org/pypi/live2d-py/json
- ✅ Live2D sample model contents (model3.json in CubismWebSamples develop): Haru has expressions F01-F08, motions Idle x2 and TapBody x4, LipSync ParamMouthOpenY. Hiyori has no expressions, Idle x9, TapBody x1, LipSync ParamMouthOpenY. Mao has exp_01-exp_08, Idle x2, TapBody x6, LipSync ParamA. Natori has named expressions Angry/Blushing/Normal/Sad/Smile/Surprised plus exp_01-05. Ren has exp_01-05. Mark, Rice and Wanko have no expressions.  
  Source: https://raw.githubusercontent.com/Live2D/CubismWebSamples/develop/Samples/Resources/<Model>/<Model>.model3.json
- ✅ Sample-model licensing. Live2D Original Characters (Hiyori, Haru, Mao Niziiro, Mark-kun, Rice, Ren Foster, Wankoromochi and others) may be used by General Users and Small-Scale Enterprises (<10M JPY) for commercial or non-commercial purposes. A copyright notice is required, e.g. 'This content uses sample data owned and copyrighted by Live2D Inc....'. Hiyori and Miara allow no design changes. Jin Natori is a Collaboration Character: non-commercial use only for general users.  
  Source: https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html (v1.6, 2025-02-03) ; https://www.live2d.com/eula/live2d-sample-model-terms_en.html
- ✅ Neuro-sama context from the research corpus (not clone facts). Her debut avatar was the stock Live2D Hiyori Momose (T4). Her Live2D body motion is a pre-recorded loop, and V2 dancing was driven by audio-visualizer data (T5). After the 2026 rewrite, Unity is a restartable render layer and the characters 'survive outside of Unity' (T1). Iteration-18 tools included 'spin her model' (T5). The claim that Neuro pipes TTS into VTube Studio via a virtual cable is clone-derived and was dropped.  
  Source: /root/.claude/uploads/02efebb2-c032-5a2c-81bb-f71e8a5461a3/a7c776ce-C08-avatar.md
- ✅ Container tests: a fake VTS server that answers once per 1/60 s frame, run with the provided client on websockets 17.1 / Python 3.11. Token auth, stored-token reuse, 120 frames of InjectParameterDataRequest at 60 Hz (120 delivered, 0 dropped), a fire-and-forget error surfacing errorID 453, an early event carrying the subscription's requestID not being mistaken for the response, and hotkey error 202 all passed. Lip-sync analysis of 30 s of 24 kHz audio took about 159 ms (pure-Python one-pole loop).  
  Source: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/avatar/test_avatar.py (run output)
- ⚠️ unverified — It is not verified that VoiceA/I/U/E/O/VoiceSilence can be injected via the API while the VTS microphone is off. The docs say any default parameter can be injected, but this was not tested against real VTS.  
  Source: inference from README 'You can feed in data for any default or custom parameter'
- ⚠️ unverified — It is not verified that VTS answers WebSocket ping frames. The client disables pings (ping_interval=None) and keeps the link alive with continuous 60 Hz injection plus an optional APIStateRequest heartbeat.  
  Source: no documentation found

## Install (Windows)

```
VTube Studio route (default):
1. Install Steam, then VTube Studio (https://store.steampowered.com/app/1325860/VTube_Studio/). Launch it once through Steam; later it can run via "start_without_steam.bat" next to the exe. If the "Remove Watermark" DLC ($14.99) is bought, VTS must be started with Steam once before the DLC works offline. The FAQ requires a paid version for monetized use.
2. In VTS settings (first tab): turn on "Start API (Allow Plugins)" and keep port 8001. If a firewall or antivirus prompts, allow localhost.
3. Load a test model.
   - Hiyori is bundled.
   - For expression tests, download Mao or Haru from https://www.live2d.com/en/learn/sample/ and copy the model's runtime folder (the one with *.model3.json, *.moc3, textures, motions, expressions) into "<Steam>\steamapps\common\VTube Studio\VTube Studio_Data\StreamingAssets\Live2DModels\<Name>\". The folder name "Live2DModels" is confirmed by VTSFolderInfoRequest; the exact Steam path is standard but was not verified here.
   - In Model Settings, run auto-setup, then check the mappings: MouthOpen → ParamMouthOpenY (ParamA for Mao) and MouthSmile → ParamMouthForm.
   - Set the smoothing slider low (0-10) for mouth parameters, because we pre-smooth.
   - Turn on Auto-Blink for the eye-open outputs and Auto-Breath for ParamBreath.
   - Set the idle animation to a *.motion3.json from the "Idle" group.
   - Leave webcam tracking OFF (the watermark only shows while it is on) and leave "Use microphone" OFF.
4. In VTS, create expression hotkeys or note the .exp3.json filenames for each emotion (Mao: exp_01..exp_08.exp3.json; Haru: F01..F08.exp3.json).
5. OBS: install the Spout2 plugin matching your OBS version from https://github.com/Off-World-Live/obs-spout2-plugin/releases. Do NOT follow the "Resize Output" step in Spout's own guide. In VTS, enable Spout2 and use the Color Picker background set to transparent black. In OBS, add a "Spout2 Capture" source with sender "VTubeStudioSpout" and Composite mode "Premultiplied Alpha".
6. Python (the project runs on Python 3.11 or 3.12 on Win11):
   py -3.11 -m venv .venv && .venv\Scripts\activate
   pip install "websockets>=17.1,<18" "numpy>=2"
   Do not install pyvts, which pulls in opencv-python.
7. First run: the plugin sends AuthenticationTokenRequest and a popup appears inside VTS; the streamer clicks "Allow". The token is saved, for example to %APPDATA%\AI_Vtube\vts_token.txt. Do this off-stream: if the token is revoked, the popup reappears.
8. Second character: click "Start VTube Studio" again in VTS, or run start_without_steam.bat a second time. It takes the next free port (8002, ...) and gets the window title "VTube Studio Window 2". Discover it via UDP 47779, and add a second Spout2 sender ("VTubeStudioSpout2").

Browser-source fallback (optional):
- Download the Cubism SDK for Web (5-r.5) from https://www.live2d.com/en/sdk/download/web/ and accept the license. Copy Core/live2dcubismcore.min.js to renderer/vendor/ and keep that path gitignored; never commit it.
- Serve renderer/ from Python at http://127.0.0.1:8765.
- Add an OBS Browser Source at http://127.0.0.1:8765/avatar.html?ws=ws://127.0.0.1:8766/avatar, 1920x1080 with the page background transparent. Uncheck "Shutdown source when not visible".
- The page uses pixi.js@7.4.3 and pixi-live2d-display-lipsyncpatch@0.5.0-ls-8. Vendor these too for offline use instead of relying on the CDN.

CI: VTS does not exist on Linux. Unit tests use the FakeVTS websocket server in test_avatar.py, which runs on the ubuntu and windows GitHub Actions runners.
```

## API notes

Reference code, all under /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/avatar/:
- vts_client.py (tested client)
- lipsync.py (envelope + IdleMotion)
- emotion.py (tag parser + expression state machine, tested)
- discovery.py (UDP 47779, tested)
- test_avatar.py (FakeVTS + end-to-end tests)
- avatar.html (browser-renderer sketch, UNTESTED)

Raw VTS docs are saved in ../vts_readme.md, ../vts_Events_README.md, ../vts_Files_ErrorID.cs, ../vts_Files_HotkeyAction.cs and ../wiki_*.md.

== ENVELOPE (every request) ==
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"<=64 ASCII","messageType":"<X>Request","data":{...}}
The response messageType is <X>Response, or "APIError" with data {errorID, message}.

== AUTH ==
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"a1","messageType":"AuthenticationTokenRequest","data":{"pluginName":"Pailin Brain","pluginDeveloper":"AI_Vtube","pluginIcon":"<base64 128x128 PNG, optional>"}}
 -> {"messageType":"AuthenticationTokenResponse","data":{"authenticationToken":"..."}}
    or APIError errorID 50 (denied) / 51 (a token request is already in progress)
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"a2","messageType":"AuthenticationRequest","data":{"pluginName":"Pailin Brain","pluginDeveloper":"AI_Vtube","authenticationToken":"..."}}
 -> {"messageType":"AuthenticationResponse","data":{"authenticated":true,"reason":"..."}}
Any other request sent before auth returns errorID 8 (RequestRequiresAuthetication; the ID is spelled that way in ErrorID.cs).

== LIP-SYNC / IDLE FRAME (60 Hz; at least 1 Hz is required) ==
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"ff-3f2a...","messageType":"InjectParameterDataRequest","data":{"faceFound":true,"mode":"set","parameterValues":[{"id":"MouthOpen","value":0.72},{"id":"MouthSmile","value":0.64},{"id":"Brows","value":0.6},{"id":"FaceAngleX","value":3.1},{"id":"FaceAngleY","value":-1.2},{"id":"FaceAngleZ","value":2.0},{"id":"EyeLeftX","value":0.1},{"id":"EyeRightX","value":0.1},{"id":"EyeOpenLeft","value":1.0},{"id":"EyeOpenRight","value":1.0}]}}
 -> {"messageType":"InjectParameterDataResponse","data":{}}
Errors: 453 unknown parameter, 454 parameter held by another plugin, 451 value invalid.
Additive nudge, e.g. a bonk: "mode":"add", which any number of plugins may use.

== EXPRESSIONS / HOTKEYS ==
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"e1","messageType":"ExpressionActivationRequest","data":{"expressionFile":"exp_03.exp3.json","fadeTime":0.3,"active":true}}
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"e2","messageType":"ExpressionStateRequest","data":{"details":false}}   // reconcile after connect / ModelLoadedEvent
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"h1","messageType":"HotkeysInCurrentModelRequest","data":{}}
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"h2","messageType":"HotkeyTriggerRequest","data":{"hotkeyID":"Wave"}}   // an ID, or a name (case-insensitive)
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"m1","messageType":"CurrentModelRequest"}
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"m2","messageType":"MoveModelRequest","data":{"timeInSeconds":0.25,"valuesAreRelativeToModel":true,"rotation":90}}   // 'spin' = four of these
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"s1","messageType":"StatisticsRequest"}   // data.framerate: use min(60, framerate) as the driver rate
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"f1","messageType":"FaceFoundRequest"}
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"x1","messageType":"ParameterCreationRequest","data":{"parameterName":"PailinBlush","explanation":"AI blush 0..1","min":0,"max":1,"defaultValue":0}}

== EVENTS ==
{"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0","requestID":"ev1","messageType":"EventSubscriptionRequest","data":{"eventName":"ModelLoadedEvent","subscribe":true,"config":{}}}
Also useful: HotkeyTriggeredEvent {onlyForAction, ignoreHotkeysTriggeredByAPI}, ModelClickedEvent (headpat reactions), TrackingStatusChangedEvent. ExpressionToggledEvent is beta-only. Event frames arrive with messageType "<Name>Event" and may carry the subscription's requestID.

== CLIENT USAGE (vts_client.py) ==
from vts_client import VTSClient, VTSAPIError
vts = VTSClient(url="ws://127.0.0.1:8001", plugin_name="Pailin Brain", plugin_developer="AI_Vtube", token_path=Path(os.environ["APPDATA"])/"AI_Vtube"/"vts_token.txt")
await vts.connect()
# inside connect: connect(url, proxy=None, compression=None, ping_interval=None, open_timeout=3, max_size=16 MiB)
assert await vts.authenticate()
# authenticate: stored token -> AuthenticationRequest; on failure, AuthenticationTokenRequest (timeout 120 s) -> save -> AuthenticationRequest
await vts.inject({"MouthOpen": 0.7, "MouthSmile": 0.6})
# pipelined: requestID prefix 'ff-', at most 8 in flight, extra frames dropped (VTS answers once per render frame); errors go to vts.last_ff_error
await vts.set_expression("exp_03.exp3.json", True, 0.3)
await vts.trigger_hotkey("Wave")
await vts.subscribe("ModelLoadedEvent")
ev = await vts.events.get()
Key rule in the reader loop: resolve a pending future ONLY if messageType == the expected '<X>Response' or 'APIError'. Everything else goes to the events queue.

== LIP-SYNC (lipsync.py) ==
mouth, form = mouth_tracks(pcm_float32_mono, sr, LipSyncConfig(fps=60, floor_db=-45, ceil_db=-12, gamma=0.7, attack_ms=30, release_ms=80, lead_ms=40))
Index k corresponds to playback time k/60 s. The one-pole coefficient is a = 1 - exp(-1000/(fps*tau_ms)), with the attack coefficient used when the target is above the current state and the release coefficient otherwise. MouthForm = clip(0.5 + 0.35*(log10(E[1.8-4 kHz]/E[250-900 Hz]) + 1)), gated to 0.5 when unvoiced and then smoothed. This is a heuristic; calibrate it on the actual Thai TTS voice.

== DRIVER LOOP (glue; the player is the audio component's object) ==
async def avatar_loop(vts, player, emo, fps=60):
    loop = asyncio.get_running_loop()
    idle = IdleMotion()
    t0 = loop.time()
    i = 0
    while True:
        seg = player.current_segment()   # has .mouth/.form arrays; None when silent or interrupted
        mo = fo = 0.0
        if seg is not None:
            k = int(player.position_s() * fps)   # from the sounddevice stream clock
            if 0 <= k < len(seg.mouth):
                mo, fo = float(seg.mouth[k]), float(seg.form[k])
        b = emo.baseline()   # {'MouthSmile':..,'Brows':..}
        v = {"MouthOpen": mo,
             "MouthSmile": min(1, max(0, b["MouthSmile"] + 0.6*(fo - 0.5)*min(1, 3*mo))),
             "Brows": b["Brows"], "EyeOpenLeft": 1.0, "EyeOpenRight": 1.0,
             **idle.sample(loop.time() - t0, mo)}
        if vts.connected:
            await vts.inject(v)
        # else: the reconnect task re-runs connect()+authenticate() with backoff
        i += 1
        await asyncio.sleep(max(0.0, t0 + i/fps - loop.time()))

== EMOTION (emotion.py) ==
emo, clean = split_emotion("[happy] สวัสดีค่า~", known={"neutral","happy","sad","angry","surprised","shy","smug"})
# -> ('happy', 'สวัสดีค่า~'); ALL [tags] are stripped before TTS
Table example for Mao:
{"neutral": EmotionSpec([], None, .5, .5),
 "happy":   EmotionSpec(["exp_03.exp3.json"], None, .9, .7),
 "sad":     EmotionSpec(["exp_05.exp3.json"], None, .15, .2), ...}
Verify which exp_NN matches which face in VTS first; Open-LLM-VTuber's default Mao map uses indices joy=3, sadness=1, anger=2, surprise=3, neutral=0. Call EmotionController.apply(emo) when the player STARTS that sentence's audio. Decay to neutral a few seconds after speech ends.

== DISCOVERY (discovery.py) ==
insts = await discover_vts(2.5)   # [{'active','port','instanceID','windowTitle','host'}]
Bind UDP 0.0.0.0:47779 with SO_REUSEADDR and parse VTubeStudioAPIStateBroadcast.

== BROWSER FALLBACK (avatar.html, untested) ==
Python pushes {"t":"params","v":{"ParamMouthOpenY":0.7,"ParamMouthForm":0.2,"ParamAngleX":3}} at 60 Hz, {"t":"expr","name":"exp_03"} and {"t":"motion","group":"TapBody","index":0}. The page applies params in model.internalModel.on('beforeModelUpdate', ...) via coreModel.setParameterValueById(id, v). Here the IDs are Live2D parameter IDs (ParamMouthOpenY, ParamAngleX, ...), not VTS input names. Map them through the model's model3.json Groups.LipSync / standard parameter list.

## Latency & resources

Python side, measured in the container:
- Envelope + form analysis costs about 5 ms per 1 s of audio (30 s in 159 ms with a pure-Python one-pole loop). Run it once per TTS segment before playback, off the event loop, e.g. with asyncio.to_thread if segments are long.
- Per-frame work is a dict build plus json.dumps of about 0.5-0.7 KB. At 60 msg/s that is about 40 KB/s on localhost, which is negligible.
- A fake server that answers once per 1/60 s frame delivered 120/120 frames with 0 drops when using pipelining (at most 8 in flight). If the client awaited every response instead, throughput would be capped at VTS's render fps and the stream would start lagging.

VTS side (documented, or inferred where noted):
- Injected values apply on the next rendered frame: up to 16.7 ms at 60 fps, or 33 ms at 30 fps. Check StatisticsResponse.framerate.
- VTS's own per-mapping smoothing slider adds lag, so keep it low for the mouth.
- Hotkeys and expressions are queued, one per 5 frames (about 83 ms at 60 fps), plus the expression fade (default 0.25 s, maximum 2 s).
- Spout2 to OBS is described as near-zero CPU. OBS adds up to one output frame (16.7 ms at 60 fps).
- Audio path: WASAPI shared mode typically adds 10-30 ms (typical figure, not measured here).
- Net effect: the video trails the audio by roughly 20-50 ms. Hence the default lead_ms is 40; tune it between 20 and 80 by eye, and/or use OBS audio Sync Offset.
- Emotion changes should be scheduled at segment playback start, not when the text is generated, so they stay aligned with speech. Neuro's historic response delay is about 700 ms, so the avatar path adds under 5% of that.

Resources:
- VTS minimum spec: DX11 GPU with 2 GB VRAM, 4 GB RAM.
- On the RTX 4070 expect a few hundred MB of VRAM for one 4K-texture model and low GPU load with trackers off. This is an unverified estimate: measure with nvidia-smi and VTS StatisticsResponse while the LLM is running.
- Risk: when llama.cpp saturates CUDA, VTS frame pacing may drop. Mitigate by capping VTS at 60 fps and watching StatisticsResponse.framerate.
- Two VTS instances (Pailin + twin) roughly double VTS's GPU/VRAM use.
- The browser-source path uses OBS's CEF GPU process instead of VTS. Cost is similar but not measured.
- live2d-py would put rendering inside our Python process. Avoid that unless it runs in a separate process.

## Pitfalls

- Do NOT route responses by requestID alone. VTS event frames can carry the subscribing request's requestID; for example, ExpressionToggledEvent with sendAllActiveStatesOnSubscription fires before or around the subscription response. Match on requestID AND the expected '<X>Response'/'APIError' messageType. The container test covers this case.
- Do NOT await every InjectParameterDataRequest response. VTS answers once per render frame, so a send-then-await loop caps at VTS fps and falls behind. Pipeline the sends, bound the number in flight, and drop frames rather than queue them (stale mouth values are worse than skipped ones).
- Injected parameters revert unless re-sent at least once per second. Keep streaming during silence (MouthOpen=0) and during LLM 'thinking'. After a reconnect, the first frames must restore state.
- Only one plugin can 'set' a parameter; a second one gets error 454. Another plugin such as 'VTS Desktop Audio' holding MouthOpen will break our lip-sync. Detect 454 and tell the operator, or fall back to a custom parameter such as 'PailinMouthOpen' mapped in the model.
- Priority trap: expressions (P4) and one-time animations (P3) override tracking/injected values (P2). Emotion .exp3.json files and reaction motions must NOT key ParamMouthOpenY/ParamA or head-angle parameters, or lip-sync and idle motion freeze while they are active. If an expression must touch ParamMouthForm, use Add mode.
- ToggleExpression hotkeys toggle, so firing 'happy' twice turns it off. Prefer ExpressionActivationRequest with explicit state, and reconcile with ExpressionStateRequest after connect or ModelLoadedEvent. ExpressionToggledEvent is public-beta only.
- Hotkey cooldown is ambiguous: the README says a given hotkey fires at most once every 5 frames, while the ErrorID.cs comment says a '5 second cooldown'. Treat hotkey errors 200/203 as non-fatal cosmetic failures and never block speech on them.
- Auth: AuthenticationTokenRequest blocks until the human clicks Allow, so use a long timeout. Sending it twice gives error 51. pluginName/pluginDeveloper must match exactly; renaming the plugin invalidates the token. If the user revokes access the popup reappears, so authenticate before going live.
- websockets>=15 auto-applies HTTP/SOCKS proxies from the environment or registry. Pass proxy=None for ws://127.0.0.1 (our dev container sets HTTPS_PROXY). websockets 17 needs Python>=3.11; pin <18. Also pass compression=None and ping_interval=None, since it is unverified whether VTS answers pings.
- Use 127.0.0.1 rather than 'localhost' to avoid IPv6 ::1 resolution delays on Windows. This is general websockets/Windows behaviour, not VTS-documented.
- pyvts 0.3.3 has no request/response matching (send then recv), no event dispatcher, pulls opencv-python, and was last released in Sep 2024. coovts 0.1.0 is days old and needs Python>=3.12. Own a ~150-line client instead.
- Licensing: a monetized stream needs a paid VTS version (FAQ). The bundled Akari model is test-only. Jin Natori is non-commercial. Hiyori and Miara allow no design changes. Streams using Live2D sample models must show the Live2D copyright notice, e.g. in the stream description.
- The browser-renderer path carries legal risk. Cubism Core is proprietary. An app that loads arbitrary models is an 'Expandable Application' (avatars / live-streaming apps), which needs Live2D approval and is not covered by the small-business exemption. Core must not be committed into an MIT repo (clause 5.3.2). Keep Core as a user-downloaded, gitignored file.
- Core and framework version mismatch: pixi-live2d-display and its forks bundle an older Cubism 4 framework and need pixi v6/v7, not pixi 8.21. Whether they work with the newest Core (06.00.x, SDK 5-r.5) is unverified. Pin a Core version you have tested. Never hotlink cubism.live2d.com in production; the lib README warns the link sometimes goes down.
- Thai-specific: strip every [tag] before TTS, or the TTS will read the brackets aloud. Do not rely on VTS uLipSync calibration for Thai vowels. Our RMS envelope is language-agnostic; the MouthForm heuristic needs tuning against the chosen Thai voice.
- Interruptions ('critical' priority speech cut, Neuro-style): the driver must read LIVE player state and position, not a precomputed schedule. Otherwise the mouth keeps moving after audio stops. The release tau (80 ms) closes the mouth naturally.
- With no face tracker, the head parameters mapped to FaceAngle* sit at neutral unless injected, and the model looks frozen. Either inject procedural motion (IdleMotion) or remove those mappings so the idle .motion3.json (P1) drives them. Pass faceFound:true so the 'tracking lost' behaviour does not trigger.
- MoveModelRequest: timeInSeconds is limited to 0-2 and rotation to ±360. While a move runs the user cannot drag the model. Implement 'spin' as 4 x 90° relative moves. The single +360 relative case was not verified.
- Multiple VTS instances auto-increment ports, so never hard-code 8002. Use UDP discovery and windowTitle. Whether one token is valid across instances is unverified; handle a re-prompt.
- VTS has no Linux build and GitHub runners have no GPU or audio. All avatar tests must use the FakeVTS server; the Windows CI runner can also run it.

## Open questions

- Can VoiceA/I/U/E/O/VoiceSilence be injected through InjectParameterDataRequest while the VTS microphone is off? The docs say 'any default parameter'; test on the real app.
- Does VTS's websocket server answer RFC 6455 ping frames? The client currently disables pings.
- Hotkey cooldown: is it 5 frames (README) or 5 seconds (ErrorID.cs comment)? Measure on VTS 1.35.10.
- Full list of models bundled with VTS beyond Akari and Hiyori.
- Actual VRAM/GPU cost of VTS (one or two instances) on the RTX 4070 while the LLM saturates CUDA, and whether frame pacing drops.
- Does faceFound:true stop VTS's tracking-lost behaviour when no tracker has ever been started in the session?
- Is one authentication token valid across multiple simultaneous VTS instances, which appear to share one StreamingAssets/Config?
- Legal: would Live2D treat an open-source AI VTuber whose browser renderer loads arbitrary models as an 'Expandable Application'? Relevant only if the browser path is published or used as the default.
- Compatibility of pixi-live2d-display-lipsyncpatch 0.5.0-ls-8 with Cubism Core 06.00.x from SDK for Web 5-r.5; pin a tested Core.
- Pailin's own model: who commissions it, which standard parameter IDs, whether it gets vowel blendshapes (ParamA/I/U/E/O) and emotion .exp3.json files that avoid mouth parameters, and a Cubism 5.x editor version compatible with VTS 1.35.
- Correct exp_NN → emotion mapping for Mao/Haru; check visually in VTS before encoding it into config.
