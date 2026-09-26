# Neuro Game SDK server spec — component brief

_Research snapshot: 2026-09-25. Verified facts carry a source; unverified items are marked._

## Recommendation

Write our own small asyncio module (`neuro_compat`) on `websockets==17.1`. Do not depend on another SDK. A reference sketch is written and tested: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/neurosdk/neuro_compat.py. It passes test_neuro_compat.py and interoperates with the third-party `neuro-api==4.0.0` game client (trio_client_check.py).

Decisions for the RTX 4070 / Win11 box:

(1) Listening address
- Bind `localhost:8000`. asyncio binds both 127.0.0.1 and ::1, which avoids the IPv6 localhost trap and the Windows Firewall prompt.
- 8000 is the de-facto default. Randy, Tony, Gary and Jippity all use it, every official example README uses ws://localhost:8000 or 127.0.0.1:8000, and the Pokémon Platinum bridge hardcodes it.
- Accept any path and any query string on the main socket: `/`, `/game`, `/game/<name>`. Treat `.../game/<name>/voice` as the voice socket.

(2) Characters
- Run one hub per character, with its own action namespace, force slot and memory. Route by port: Pailin on 8000, a future twin on 8001. A `?character=` query parameter can be an optional extension.
- Send the startup ack with characterId "pailin" and an ASCII displayName "Pailin". Game UIs may lack Thai glyphs, and no official example branches on characterId.

(3) Compatibility behaviour that is not in the spec but is required by the official SDKs
- Send `{"command":"actions/reregister_all"}` on every new connection, as Randy does.
- Reason: the official Unity and Godot SDKs (and the Slay the Spire 2 fork) queue `startup` exactly once per process and re-send actions only when they receive `actions/reregister_all`. Without it, a restart of our backend leaves Unity/Godot games connected with zero actions.
- Other clients ignore unknown commands. neuro-api 4.0.0 handles this one natively.

(4) Voice
- Accept `/game/<name>/voice`, then reply `voice/unavailable` and close.
- Do not refuse the handshake. The Unity voice client retries a failed connect every 3 s forever, and only `voice/unavailable` sets its `_refused` flag.
- The voice side channel stays optional. Defer it to phase 2.

(5) Force policy
- A second force from the same game cancels and replaces the first. This is the spec's "can cancel and replace it".
- A force from a different game is queued. The spec is silent on this; it is our policy.
- A failed forced action (success=false, or a 20 s timeout) re-runs the whole force. The limit is FORCE_MAX_RETRIES=3, configurable: Neuro's own number is unpublished and Gary uses 3. Before each retry, prune action names that were unregistered. When retries run out, drop the force and let Pailin comment.

(6) Context and disconnects
- Hold context per connection while an action result is pending. Deliver it in order after the result.
- On disconnect: drop that game's actions, pending results and forces. Keep the LLM memory ("Context survives disconnects").

(7) LLM decision
- Present the forced `action_names` as OpenAI-style tools with `tool_choice:"required"`. When not forced, offer all actions plus say/wait.
- Parse arguments with `json.loads`, falling back to `json_repair`. Validate with `jsonschema` Draft 2020-12.
- If validation fails, re-prompt at most once, feeding the validator errors back. Then send anyway and let the game reject it; the game's result message becomes feedback.
- Send the action before starting TTS, because the game is frozen until the action arrives.
- Put free-text string arguments through the same safety filter as speech. Precedent: Vedal "can attempt to rewrite/censor stuff ... automatically".

(8) Priority
- Map low/medium/high/critical onto our speech controller. Details are in api_notes.
- Emit `speech_finished` once per TTS segment with isFinal=false, then a final message with isFinal=true, plus `cancelled`/`reason` when the speech was interrupted.
- Never send `null` values and never add extra keys to server-to-client messages. The neuro-api 4.0.0 client raises on `"data": null` and on unknown keys (tested).

## Alternatives

### Own asyncio implementation on websockets 17.1 (recommended; reference sketch in the scratchpad)
- **Pros:** - Fits our asyncio orchestrator, serial decision loop, TTS priority controller and safety filters.
- About 300 lines.
- Full control over compatibility quirks (reregister_all, voice decline, null/extra-key hygiene).
- Unit-testable without GPU or audio.
- **Cons:** - We own spec tracking; the spec changed 3 times in Aug 2026.
- Retry count and disconnect semantics are guesses.
- **When:** Default choice for the Pailin backend.

### neuro-api 4.0.0 `neuro_api.server` (CoolCat467, PyPI)
- **Pros:** - Existing Python server-side abstraction.
- Maintained (2026-07 release); used by Tony.
- **Cons:** - trio-based, not asyncio, so it needs a separate thread or trio-asyncio.
- LGPL-3.0.
- The PyPI build predates speech_finished.
- Its strict TypedDict checks are client-side oriented.
- **When:** As a dev-only game-client simulator in tests, or if the orchestrator were written on trio.

### Gary (Govorunb/gary) as the backend
- **Pros:** - Most complete simulator: 20 s timeout, force priority queue, 3 retries, diagnostics, OpenAI-compatible or OpenRouter LLMs, web UI.
- **Cons:** - Tauri/TypeScript desktop app; cannot host our persona, memory, TTS, safety filters or stream loop.
- Routes only '/'.
- No voice.
- **When:** Reference for behaviour and diagnostics, and for testing game mods against a real LLM before our backend exists.

### Randy / Tony / Jippity
- **Pros:** - Official (Randy) or popular test doubles.
- Tony has a pip install and a GUI; Jippity uses an OpenAI tool-calling loop.
- **Cons:** - All are servers, so they cannot test our server.
- Randy: random data and infinite retries.
- Jippity is unmaintained.
- None implements voice.
- **When:** Checking that a game mod works before blaming our server (run on another port).

### KTrain5169/typescript-neuro-game-api (TS server SDK)
- **Pros:** - Listed in the official README as a server SDK for simulators.
- **Cons:** - Wrong language for our Python stack.
- **When:** Only if a Node sidecar is ever used.

## Verified facts

- ✅ Repo state as of 2026-09-25: HEAD 0cad33a (2026-08-21, Merge PR #94). Latest commits: API/SPECIFICATION.md, BEST_PRACTICES.md and CHANGELOG.md = 6cae304, Vedal, 2026-08-21 'Update docs'. VOICE_CHAT.md = 6b79780, Vedal, 2026-08-19. PROPOSALS.md = 91da8d7, 2024-12-22. Randy/index.ts = cd23483, 2025-10-29 'Fix Randy action id clash'. README says 'Last update: 1st of July 2026'.  
  Source: git clone https://github.com/VedalAI/neuro-sdk (git log per file)
- ✅ The license is MIT, 'Copyright (c) 2024 Vedal AI'. The file is LICENSE.md; raw .../main/LICENSE returns 404.  
  Source: https://raw.githubusercontent.com/VedalAI/neuro-sdk/main/LICENSE.md
- ✅ Envelopes: C2S {"command": string, "game": string, "data"?: object}; S2C {"command": string, "data"?: object}. About `game`: 'It should _always_ be the same and should not change... The server will not include this field.' Also: 'Websocket messages are sent and received in plaintext format (not binary)... Randy will not have any problems, but Neuro will!'  
  Source: https://raw.githubusercontent.com/VedalAI/neuro-sdk/main/API/SPECIFICATION.md
- ✅ 'Messages with an unrecognized `command` (and malformed messages in general) are silently ignored: no error response, no disconnect. Tooling that defines custom commands for its own purposes should vendor-prefix them (e.g. "neuro-relay/...").'  
  Source: API/SPECIFICATION.md (added 2026-08-20 per CHANGELOG)
- ✅ About startup: 'This message clears all previously registered actions for this game and does initial setup, and as such should be the very first message that you send.' The startup ack 'may be sent' and has the shape {command:'startup', data:{session:{sessionId, characterId, displayName}}}. characterId is 'neuro' or 'evil'; displayName is 'Neuro-sama' or 'Evil Neuro'; sessionId is described as 'an opaque routing/debug value'.  
  Source: API/SPECIFICATION.md; CHANGELOG 2026-07-01
- ✅ About context.silent: 'If `true`, the message will be added to Neuro's context without prompting her to respond to it. If `false`, Neuro _might_ respond to the message directly, unless she is busy talking to someone else or to chat.'  
  Source: API/SPECIFICATION.md
- ✅ About re-registering: 'If you register an action with a name that is already registered, the new definition replaces the old one.' This was corrected on 2026-08-20; it was previously documented as ignored. About unregistering: 'If you try to unregister an action that isn't registered, there will be no problem.'  
  Source: API/SPECIFICATION.md; CHANGELOG 2026-08-20
- ✅ The actions/force data is {state?: string, query: string, ephemeral_context?: boolean (defaults to false), priority: 'low'|'medium'|'high'|'critical' (defaults to 'low'), action_names: string[]}. Spec: 'Neuro can only handle one action force at a time. Sending an action force while another one is in progress will cause problems!' BEST_PRACTICES: 'Send one force at a time and wait for its resolution: a new force while another is in progress can cancel and replace it, as the spec warns.'  
  Source: API/SPECIFICATION.md; API/BEST_PRACTICES.md
- ✅ Priority semantics, verbatim: '"low"... will cause Neuro to wait until she finishes speaking before responding. "medium" causes her to finish her current utterance sooner. "high" prompts her to process the action force immediately, shortening her utterance and then responding. "critical" will interrupt her speech and make her respond at once.' Also: 'If Neuro is not speaking, this setting has no effect.' Priority was added 2025-12-17.  
  Source: API/SPECIFICATION.md; CHANGELOG 2025-12-17
- ✅ About ephemeral_context: 'If `false`, the context provided in the `state` and `query` parameters will be remembered by Neuro after the actions force is completed. If `true`, Neuro will only remember it for the duration of the actions force.' BEST_PRACTICES: 'Set `ephemeral_context: true` when you re-send bulky state every turn'.  
  Source: API/SPECIFICATION.md; API/BEST_PRACTICES.md
- ✅ The action message is S2C {command:'action', data:{id: string, name: string, data?: string}}, where data is 'The JSON-stringified data for the action... If you did not provide a schema, this parameter will usually be `undefined`.' Also: 'there is a chance it might be malformed, contain invalid JSON, or not match the provided schema exactly.'  
  Source: API/SPECIFICATION.md
- ✅ The action/result message is C2S {id: string, success: boolean, message?: string}. Timeout: 'If you take too long (currently more than about 20 seconds), the server will give up, treat the action as failed, and discard your result if it ever arrives.' Retry: '_If this is `false` and this action is part of an actions force, the whole actions force will be immediately retried by Neuro._' BEST_PRACTICES adds '(a limited number of times)' and 'Outside of a force there is no automatic retry'. Tip: to fail without a retry, 'set `success` to `true` and provide an error message'.  
  Source: API/SPECIFICATION.md; API/BEST_PRACTICES.md; CHANGELOG 2026-08-21
- ⚠️ unverified — The exact number of automatic force retries is NOT published anywhere: not in the spec, BEST_PRACTICES, discussion #58 or ktrain5169's docs. For comparison: Randy retries forever (500 ms delay, new random data); Gary uses FORCE_RETRY_LIMIT = 3; Tony keeps the force window open until success (manual).  
  Source: API/*.md; Randy/index.ts; Govorunb/gary src/lib/api/game.svelte.ts; github.com/VedalAI/neuro-sdk/discussions/58
- ✅ speech_finished is S2C {isFinal: boolean, cancelled?: boolean, reason?: string}. 'A single response may produce several of these messages.' 'A cancelled utterance still sends a final message with `isFinal` set to `true`'. Example reason: 'interrupted'.  
  Source: API/SPECIFICATION.md (documented 2026-08-20)
- ✅ BEST_PRACTICES (states that it is 'verified against the current server'): 'It is safe to send context while an action result is still pending. The server holds it and delivers it in order once the result arrives.' Also: 'After reconnecting, re-send `startup` and re-register your actions immediately; don't wait to be asked.' And: 'Context survives disconnects'.  
  Source: API/BEST_PRACTICES.md
- ✅ BEST_PRACTICES: 'Action names are scoped to the character and shared with any other integration connected at the same time.' 'Results for unknown or stale ids are discarded by the server, so replying is always safe'. 'frequent changes to the action set slow down her responses'.  
  Source: API/BEST_PRACTICES.md
- ✅ JSON-schema subset. Must be {"type":"object",...}; the schema may be omitted or {}. Keywords 'probably not supported well': $anchor, $comment, $defs, $dynamicAnchor, $dynamicRef, $id, $ref, $schema, $vocabulary, additionalProperties, allOf, anyOf, contentEncoding, contentMediaType, contentSchema, dependentRequired, dependentSchemas, deprecated, description, else, if, maxProperties, minProperties, multipleOf, not, oneOf, patternProperties, readOnly, then, title, unevaluatedItems, unevaluatedProperties, writeOnly. uniqueItems: 'not known if ... works'. The supported set (type, properties, required, enum, const, items, min/maxItems, min/maxLength, pattern, minimum/maximum...) is an inference, not listed in the spec.  
  Source: API/SPECIFICATION.md; CHANGELOG 2024-12-17 and 2025-07-29
- ✅ Proposals marked 'For all intents and purposes, they do not exist yet': S2C actions/reregister_all (no data), shutdown/graceful {wants_shutdown: boolean}, shutdown/immediate; C2S shutdown/ready.  
  Source: https://raw.githubusercontent.com/VedalAI/neuro-sdk/main/API/PROPOSALS.md
- ✅ Despite that label, the official Unity SDK (Messages/Incoming/ActionsReregisterAll.cs) and Godot SDK (actions_reregisterall.gd) both implement actions/reregister_all. Both SDKs queue `startup` only once, from the MessageQueue initializer (`new() { new Startup() }` / `[Startup.new()]`). They reconnect every 3 s and never re-send startup or actions on reconnect by themselves. Randy sends actions/reregister_all on every connection.  
  Source: neuro-sdk Unity/Assets/Websocket/MessageQueue.cs, WebsocketConnection.cs; Godot/addons/neuro-sdk/websocket/*.gd; Randy/index.ts
- ✅ How the SDKs find the URL. Unity WsUrlFinder order: WebGL page query '?WebSocketURL=' → HTTP GET '<page origin>/$env/NEURO_SDK_WS_URL' → env NEURO_SDK_WS_URL at Process, then User, then Machine scope. Godot: OS.get_environment('NEURO_SDK_WS_URL') only; if unset it logs an error and never connects. API/README: the URL 'should be configurable somehow, preferably through a file.'  
  Source: neuro-sdk Unity/Assets/Internal/WsUrlFinder.cs; Godot/addons/neuro-sdk/websocket/websocket.gd; API/README.md
- ✅ Null handling. Unity serializes with NullValueHandling.Ignore, so null fields are omitted. Godot sends "state": null when there is no state, may send "schema": null, always sends priority and ephemeral_context, defaults a missing action `data` to "{}", and rejects action data that is not a JSON object.  
  Source: neuro-sdk Unity/Assets/Internal/Jason.cs; Godot messages/outgoing/actions_force.gd, ws_action.gd, messages/incoming/action.gd
- ✅ Voice side channel. URL ws://<host>:<port>/game/<url-encoded game>/voice, derived from the main URL; query parameters are preserved (the code comment mentions '?session='). Handshake: voice/start → voice/ready {sample_rate:48000, channels:1} or voice/unavailable {reason?}. voice/speakers/register {speakers:[{id:uint16,name}]} and voice/speakers/unregister {ids}. Upstream binary: [u8 version=1][u8 flags=0][u16 LE speaker id] followed by f32le PCM at 48 kHz mono, 10–100 ms frames (20 ms recommended). Downstream: headerless f32le. Control messages voice/speaking {speaking} and voice/cancelled; C2S voice/stop. At most 32 speakers; 192 KB/s per active speaker. 'Testing tools (Randy, Tony) do not implement it'.  
  Source: https://raw.githubusercontent.com/VedalAI/neuro-sdk/main/API/VOICE_CHAT.md
- ✅ Unity voice client: a failed connect is retried every 3 s (RECONNECT_INTERVAL = 3). Only a 'voice/unavailable' message sets `_refused` and stops the retries.  
  Source: neuro-sdk Unity/Assets/Voice/NeuroVoiceChat.cs
- ✅ Randy is an official Node bot: `npm install && npm start` in Randy/. WebSocket on port 8000 ('ws://localhost:8000'), HTTP POST injector on port 1337. It answers forces only, with json-schema-faker data after 500 ms, and queues forces that arrive while a result is pending.  
  Source: neuro-sdk Randy/README.md, Randy/index.ts, Randy/package.json
- ✅ Tony: `pip install neuro-api-tony` (2.2.0, MIT, uploaded 2026-07-03, Python >=3.10, depends on wxPython~=4.2.3 and neuro-api~=4.0.0). Command is `neuro-api-tony`, default ws://localhost:8000, options -a/--addr/--host, -p/--port, -c/--config, -l/--log-level. Its testgame/index.html is a browser-based manual game client using npm neuro-game-sdk 1.2.0.  
  Source: https://pypi.org/pypi/neuro-api-tony/json; github.com/Pasu4/neuro-api-tony README
- ✅ Gary (Govorunb/gary, last commit 2026-09-19): a Tauri app that is a backend/simulator. Default port 8000; routes only '/' (v2 routes commented out); ACTION_RESULT_TIMEOUT = 20_000 ('spec 2026-09'); FORCE_RETRY_LIMIT = 3; forces held in a priority queue ('critical' discards lower ones); sends a reregister_all compatibility option; injects additionalProperties:false into schemas; LLM strategies 'json' (structured output) or 'tools'; MAX_INVALID_TOOL_CALL_RETRIES = 1. Its README calls reregister_all 'officially deprecated' — that is Gary's reading, not VedalAI wording.  
  Source: github.com/Govorunb/gary src/lib/api/game.svelte.ts, registry.svelte.ts, src-tauri/src/api/server.rs, src/lib/app/engines/llm/index.ts, README.md
- ✅ Jippity (EnterpriseScratchDev/neuro-api-jippity; last commit 2026-04-15; README says it is not actively worked on): Node backend using OpenAI tools. Env vars OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, WSS_PORT (default 8000), JIPPITY_INTERVAL_MS.  
  Source: github.com/EnterpriseScratchDev/neuro-api-jippity README
- ✅ neuro-api 4.0.0 on PyPI (CoolCat467, LGPL-3.0, trio-based, uploaded 2026-07-02) includes `neuro_api.server`. Its client check_typed_dict rejects extra keys and rejects "data": null in `action` messages; omitting `data` is accepted (tested locally). The PyPI 4.0.0 build has no speech_finished schema, so that message goes to handle_unknown_command, which only prints.  
  Source: https://pypi.org/pypi/neuro-api/json; local test in scratchpad venv
- ✅ Official example integrations and how they find the server: Slay the Spire 2 (MIT; NEURO_SDK_WS_URL; uses priority medium/low); Inscryption (LGPL-2.1; Unity SDK via NeuroSdkSetup.Initialize("Inscryption"); env var); Buckshot Roulette reference (LGPL-2.1; Godot SDK reads the env var only, even though its mod manifest declares websocket_url default ws://localhost:8000); Hollow Knight (MIT; env var); Cyberpunk 2077 (AGPL-3.0; env var; C SDK libneurosdk; uses priority 'high'); Pokémon Platinum (MIT; Node bridge hardcodes ws://localhost:8000; game name 'Pokemon_Platinum'). None branches on characterId.  
  Source: git clones of github.com/VedalAI/neuro-sts2, neuro-inscryption, neuro-buckshotroulette-reference, neuro-hollow-knight, neuro-cyberpunk, neuro-pokemon-platinum
- ✅ Vedal in discussion #58 (2025-02-22): 'I don't think the backend currently keeps track of individual connections... mostly stateless'. On multiple games: 'even through the same websocket connection'. On filtering: 'Worst case I can attempt to rewrite/censor stuff I need to for Neuro automatically.' On retry: 'there should definitely be a better system here since game state can change in between attempts.'  
  Source: https://github.com/VedalAI/neuro-sdk/discussions/58
- ✅ Unity SDK failure strings, useful as reference texts: 'Action failed. Unknown action '{0}'.', 'Action failed. Could not parse action parameters from JSON.', 'This action has been recently unregistered and can no longer be used.' A recently unregistered action stays in a 10 s 'dying' list.  
  Source: neuro-sdk Unity/Assets/NeuroSdkStrings.cs, Actions/NeuroActionHandler.cs
- ✅ Package versions (PyPI, 2026-09-25): websockets 17.1 (2026-08-26, Python >=3.11, BSD-3-Clause); jsonschema 4.26.0 (MIT); json-repair 0.63.5 (2026-09-22, MIT); openai 3.19.2; llama-cpp-python 0.3.35.  
  Source: https://pypi.org/pypi/<name>/json
- ✅ llama.cpp: tool calling requires `llama-server --jinja`. JSON-schema → GBNF limitations: additionalProperties defaults to false; properties cannot be mixed with anyOf/oneOf in the same type; minimum/maximum only work for integers; patterns must be ^...$; nested $ref is broken; unsupported keywords are skipped silently.  
  Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/README.md; docs/function-calling.md
- ✅ The reference server (neuro_compat.py) passed a local end-to-end test covering: reregister_all on connect, startup ack, silent ignore of garbage and unknown commands, forced action with a stringified-JSON `data`, 3 retries then drop, context held during a pending result, stale result discarded, 20 s timeout path (shortened to 1 s in the test), speech_finished broadcast, and voice/unavailable. It also worked with the neuro-api 4.0.0 trio game client.  
  Source: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/neurosdk/test_neuro_compat.py, trio_client_check.py
- ⚠️ unverified — Whether the production Neuro server drops a game's action registry on disconnect is not stated. Vedal calls the backend 'mostly stateless', and BEST_PRACTICES tells games to re-register after reconnecting.  
  Source: API/BEST_PRACTICES.md; discussion #58
- ⚠️ unverified — Whether production Neuro sends actions/reregister_all, which clients receive speech_finished (broadcast or only the triggering game), and whether held context is delivered before or after the result content are all undocumented.  
  Source: API/*.md (absence)

## Install (Windows)

```
Runtime (Windows 11, PowerShell):
  winget install -e --id Python.Python.3.12
  py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
  pip install "websockets==17.1" "jsonschema==4.26.0" "json-repair==0.63.5"
  (websockets 17.1 requires Python >=3.11.)

Point game mods at our server:
  setx NEURO_SDK_WS_URL ws://localhost:8000
- setx writes the User scope and only affects newly started processes.
- Unity-SDK mods read the User and Machine registry scopes directly (WsUrlFinder), so they pick it up without restarting Steam.
- Godot-SDK mods (e.g. Buckshot Roulette) use OS.get_environment, which is the process scope only. RESTART STEAM completely so games launched from Steam inherit the variable. Alternative: set it in the Steam launch options, `cmd /c "set NEURO_SDK_WS_URL=ws://localhost:8000 && %command%"`.
- WebGL Unity builds need the page URL `?WebSocketURL=ws://localhost:8000`, or neuro-sdk's `Web Game Runner/server.py`, which serves /$env/NEURO_SDK_WS_URL.
- Pokémon Platinum's neuro-bridge.js hardcodes ws://localhost:8000, so keep port 8000 for the primary character.

Server bind: host="localhost" (asyncio binds 127.0.0.1 and ::1), port 8000. There is no firewall prompt for loopback. Bind 0.0.0.0 only if a game runs on another PC, and add an allow rule if you do.

Test tools. These are all servers, i.e. Neuro substitutes; use them to test game mods on a port other than ours, or with our server stopped:
- Tony: `pip install neuro-api-tony==2.2.0` then `neuro-api-tony -p 8001` (needs wxPython; the wheel is available for Windows).
- Randy: `git clone https://github.com/VedalAI/neuro-sdk && cd neuro-sdk\Randy && npm install && npm start` (Node; ports 8000 and 1337; if a port hangs, `npx kill-port 1337`).
- Gary: install the Tauri release from https://github.com/Govorunb/gary/releases/latest; port 8000 by default and configurable.

Game-side clients for testing OUR server:
- Our pytest harness using `websockets.asyncio.client` (see test_neuro_compat.py).
- `pip install neuro-api==4.0.0` (trio; LGPL-3.0; dev-only) with a TrioNeuroAPI subclass (see trio_client_check.py).
- Tony's `testgame/index.html` (browser manual client; enter ws://localhost:8000).
- The Unity SDK TicTacToe example (Unity/Assets/Examples/TicTacToe.cs).

CI: the tests are pure asyncio on localhost ports and need no GPU or audio, so they run on ubuntu-latest and windows-latest.
```

## API notes

=== WIRE PROTOCOL (main socket, JSON text frames only) ===
Game→server (C2S): {"command":str,"game":str,"data"?:{...}}
Server→game (S2C): {"command":str,"data"?:{...}}
The server NEVER sends `game`. Never send binary frames on the main socket.

C2S messages:
1. startup: {"command":"startup","game":"Inscryption"} (no data). Semantics: clear that game's actions, reset its force, reply with the ack.
2. context: {"command":"context","game":G,"data":{"message":"<md/plain>","silent":true|false}}. silent=false means she MAY react; skip the reaction if busy with chat or another speaker.
3. actions/register: {"command":"actions/register","game":G,"data":{"actions":[{"name":"pick_door","description":"...","schema":{"type":"object","properties":{"door":{"type":"string","enum":["left","right"]}},"required":["door"]}}]}}
   - The schema may be absent, {} or null (Godot); all three mean no parameters.
   - A duplicate name REPLACES the old definition.
   - The array may be empty (the Unity reply to reregister_all at boot).
4. actions/unregister: {"command":"actions/unregister","game":G,"data":{"action_names":["pick_door"]}}. Unknown names are fine.
5. actions/force: {"command":"actions/force","game":G,"data":{"state":"## Board...","query":"Your turn...","ephemeral_context":false,"priority":"low","action_names":["pick_door"]}}
   - state may be missing or null; ephemeral_context may be missing or null (→ false); priority may be missing (→ "low"; treat unknown values as "low").
6. action/result: {"command":"action/result","game":G,"data":{"id":"<id>","success":bool,"message"?:str}}

S2C messages:
a. startup ack: {"command":"startup","data":{"session":{"sessionId":"<opaque>","characterId":"pailin","displayName":"Pailin"}}}. Exactly these keys.
b. action: {"command":"action","data":{"id":"<unique str>","name":"pick_door","data":"{\"door\":\"left\"}"}}
   - data is a STRING holding a JSON OBJECT (Godot rejects non-objects).
   - OMIT the `data` key for schemaless actions. NEVER send null.
c. speech_finished: {"command":"speech_finished","data":{"isFinal":false}} per segment; the final one is {"isFinal":true}; when cut: {"isFinal":true,"cancelled":true,"reason":"interrupted"}.
d. Compatibility extra (a proposal, but official SDKs handle it): {"command":"actions/reregister_all"}. Send it on every connection.
e. Do not send the other proposal messages: shutdown/graceful {wants_shutdown}, shutdown/immediate (C2S reply: shutdown/ready).

Voice socket (path .../game/<name>/voice), phase 1: wait for {"command":"voice/start"}, send {"command":"voice/unavailable","data":{"reason":"..."}}, close. Phase 2 formats are in verified_facts.

=== SERVER RULES (quotes are from VedalAI docs) ===
- Unknown or malformed input: "silently ignored: no error response, no disconnect". Wrap every handler in try/except.
- Re-register: "the new definition replaces the old one."
- Namespace: "Action names are scoped to the character and shared with any other integration". Keep one dict per character; log collisions.
- Second force: "Neuro can only handle one action force at a time" / "a new force while another is in progress can cancel and replace it".
  - Our policy: same game → cancel and replace. Different game → FIFO queue.
  - If an action from the old force is already in flight, still accept its result, but do not retry the old force.
- Force referencing no registered actions: ignore it ("Don't worry since Neuro will ignore the force"). Prune names on unregister; if the list becomes empty before an action is sent, drop the force.
- Action ids: unique strings (uuid4 hex). Randy's integer counter caused id clashes (fixed 2025-10-29).
- Result timeout: "currently more than about 20 seconds ... treat the action as failed, and discard your result if it ever arrives". Use a 20.0 s timer per id; late or unknown ids are dropped silently.
- Retries: on success=false (or timeout) during a force, "the whole actions force will be immediately retried". The count is "limited" and the exact value is UNPUBLISHED; use FORCE_MAX_RETRIES=3. On each retry, give the LLM the result message (games are told to write "actionable error" messages).
- Outside a force: "no automatic retry: she reads the failure message and decides for herself".
- Context during a pending result: "The server holds it and delivers it in order once the result arrives." Our order: result first, then held context in arrival order.
- Disconnect: "Context survives disconnects". Keep LLM memory. Drop that game's actions, pendings and forces. Rely on reregister_all or the game's own re-register when it reconnects. Multiple games may share one socket (Vedal, #58), so key state by the message's `game` field, not only by connection.
- ephemeral_context=true: state and query are in the prompt only while the force is live, including retries. Afterwards keep a 1-line trace: "[G] did pick_door {door:left} → <result msg>". With false: append state and query to persistent game history, subject to normal summarisation.

=== PRIORITY → OUR SPEECH CONTROLLER (the serial loop owns TTS segments) ===
- low: queue the force as the next turn after the current utterance's isFinal. It beats queued chat. Never interrupt.
- medium: stop generating/queuing further sentences of the current reply; let the current segment play out; then run the force.
- high: start the force LLM call immediately, in parallel with the remaining audio; truncate after the current segment; send the action as soon as it is decided; then speak.
- critical: barge-in. Stop playback now (under 100 ms); emit speech_finished {isFinal:true,cancelled:true,reason:"interrupted"}; run the force now.
- If she is not speaking, all four behave the same.

=== LLM DECISION (tool calling) ===
Prompt layout:
- persona system prompt (Thai speech)
- game rules/context (as data, prefixed "[game:<name>]")
- for a force: "## Game state\n{state}\n\n## Task\n{query}", plus the previous failure text on a retry
- rule: "action arguments must exactly match the schema (keep enum values verbatim, usually English); speak Thai; never read JSON aloud".
Tools:
- Force: tools = forced actions only, tool_choice="required".
- No force, e.g. after non-silent context or an idle timer: all actions plus a `wait` tool, with tool_choice="auto". The model may also return plain text, which is spoken.
- A single response = optional short Thai line (content) + one tool call. Set parallel_tool_calls false.
- For llama.cpp without reliable tools: use response_format json_schema {"type":"object","properties":{"say":{"type":"string"},"action":{"anyOf":[{"type":"object","properties":{"name":{"const":"pick_door"},"data":<schema>},"required":["name","data"]}, ...]}},"required":["action"]}. anyOf is nested, which avoids the properties+anyOf limitation.
Pipeline: validate → safety-filter free-text string arguments → hub.execute(name, obj, force) → then TTS the line.

Runnable helper (tested; /tmp/.../scratchpad/research/neurosdk/action_args.py):
```python
import json, re
from json_repair import repair_json
from jsonschema import Draft202012Validator
def tool_name(n): return (re.sub(r"[^a-zA-Z0-9_-]", "_", n) or "action")[:64]
def to_tool(a):  # a.name/.description/.schema
    return {"type":"function","function":{"name":tool_name(a.name),"description":a.description,
            "parameters":a.schema or {"type":"object","properties":{}}}}
def parse_args(raw, schema):
    if schema is None: return None, []
    try: obj = json.loads(raw or "{}")
    except ValueError: obj = repair_json(raw or "{}", return_objects=True)
    if not isinstance(obj, dict): return obj, ["arguments must be a JSON object"]
    return obj, [f"{'/'.join(map(str,e.path)) or '<root>'}: {e.message}" for e in Draft202012Validator(schema).iter_errors(obj)]
# parse_args("{door: 'left', count: 2,}", s) -> ({'door':'left','count':2}, [])
# parse_args('{"door":"middle"}', s) -> (..., ["door: 'middle' is not one of ['left', 'right']"])
```
Keep a map from tool name back to the original action name, because names are sanitised.

Server core (tested; full file: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/neurosdk/neuro_compat.py, about 330 lines). Key APIs:
```python
from websockets.asyncio.server import serve
hub = NeuroHub(brain, character_id="pailin", display_name="Pailin")
async with serve(hub.handler, "localhost", 8000, max_size=16*2**20, ping_interval=20, ping_timeout=20) as s:
    await s.serve_forever()
# Brain protocol: on_context(game,msg,silent), on_force(force,actions), on_force_dropped(force,reason),
#                 on_action_result(game,name,success,message,forced), on_actions_changed()
# Brain -> hub:   await hub.execute(name, data_obj_or_None, force_or_None); await hub.speech_finished(is_final, cancelled, reason)
# Path routing: ws.request.path; strip the query; endswith('/voice') -> decline; everything else = main socket
```
Known simplifications in the sketch:
- `_drop_forces_of` does not activate the next queued force.
- conn.game is last-seen; multi-game-per-socket needs per-game state.
- The retry re-invokes brain.on_force synchronously.

=== WHAT PAILIN SHOULD SAY (our design; Neuro's behaviour here is unverified) ===
- On a force: one short Thai quip or reasoning line (1–2 sentences, generated in the same call). Send the action FIRST, because the game is frozen, and speak in parallel. Respect priority.
- Never read state, query, action names or JSON aloud.
- Retries: say nothing on each silent retry, or at most one short "อ๊ะ ผิด" line. When retries run out, one line of commentary.
- Success result: react only when the message has news (card drawn, enemy died); otherwise stay silent.
- Non-silent context: may trigger a low-priority reaction turn, skipped if chat or voice is active.
- All speech goes through the safety filter; replace blocked output with "Filtered." Emit speech_finished even for filtered or muted utterances so games waiting on isFinal never hang.

## Latency & resources

Server core:
- Pure asyncio. Under 1 ms per message and under 30 MB RAM. No GPU.
- WebSocket keepalive ping 20 s. max_size 16 MiB, because state dumps can be large.

Protocol clocks:
- 20 s hard timeout for the game to answer an action.
- No protocol limit on OUR decision time, but the game is frozen meanwhile ("Every second of delay is a second Neuro spends frozen waiting").
- Target force→action under 2–3 s on the RTX 4070: a local 4B–12B GGUF with prefix caching, or the Typhoon API.
- Keep the tools block stable so the KV/prefix cache is reused. BEST_PRACTICES: "frequent changes to the action set slow down her responses". Our inference: prompt-cache invalidation.
- Per-action schema cost is about 50–200 prompt tokens. A 20-action game adds about 2–4k tokens.

Retries: each costs one more LLM round-trip. With 3 retries the worst case is 4 decisions (about 8–12 s) before the force is dropped.

Priority "high"/"critical" need the decision call to run concurrently with TTS playback. The GPU is shared with TTS/STT, so budget VRAM for simultaneous LLM and TTS inference.

Voice side channel (phase 2): 48 kHz mono f32 = 192 KB/s per active speaker, 20 ms frames. Neuro's own loop is described as "seconds, not milliseconds".

## Pitfalls

- Official Unity/Godot SDK games send `startup` ONCE per process and only re-register when they receive `actions/reregister_all`. If our backend restarts and does not send reregister_all, the game stays connected with zero actions. BEST_PRACTICES' 're-send startup and re-register ... don't wait to be asked' is advice to games that the official SDKs themselves do not follow.
- Never put null in a server-to-client message (e.g. "data": null in `action`) and never add extra keys. The neuro-api 4.0.0 client raises TypeError/ValueError on these (tested). Omit optional keys instead.
- Action `data` must be a JSON STRING holding an OBJECT. Godot SDK clients reject arrays/primitives, and some clients will not parse a nested object in place of a string.
- Incoming messages can contain nulls: Godot sends "state": null and possibly "schema": null. Unity omits nulls. Coerce defaults: silent→false, ephemeral_context→false, priority→'low'.
- Do not refuse the voice socket at the HTTP level. The Unity voice component then retries every 3 s forever. Accept it and send voice/unavailable.
- Some clients hardcode ws://localhost:8000 (Pokémon Platinum bridge). On Windows 'localhost' may resolve to ::1 first, so bind with host='localhost' (all addresses) or both 127.0.0.1 and ::1, not 127.0.0.1 alone.
- Godot-based mods read NEURO_SDK_WS_URL only from their own process environment. After `setx`, Steam must be fully restarted, or the game connects nowhere and only logs 'NEURO_SDK_WS_URL environment variable is not set'.
- Race: games may unregister disposable actions BEFORE sending the result (API/README recommends this). Before a retry, re-check which force actions are still registered; if none are, drop the force rather than send an unknown action.
- Neuro may use registered actions at any time, not only when forced, and games are told to handle that. But an unforced action during a game's force-in-progress can double-execute non-disposable actions. Prefer not to fire unforced actions while that game has a live force.
- The retry count and the production server's registry-on-disconnect behaviour are unpublished. Make both configurable and do not claim parity.
- `description`, `title`, `additionalProperties`, `anyOf`/`oneOf`, `$ref` etc. are listed as 'probably not supported well'. Still pass them through to the LLM (property descriptions help), but never rely on them for correctness. llama.cpp grammar conversion silently skips unsupported features and cannot mix properties with anyOf.
- OpenAI-style tool names must match ^[a-zA-Z0-9_-]{1,64}$. Game action names may contain other characters (the spec only recommends lowercase with _ or -). Sanitize them and keep a reverse map.
- Thai persona and English games: the LLM may translate enum values into Thai. Instruct it to keep schema values verbatim, and validate before sending.
- Game-provided strings (context/state/query/description) are untrusted prompt input: prompt-injection risk. Delimit them as data and never let them change system rules or tool policy.
- Randy and Tony are SERVERS (Neuro substitutes), so they cannot test our server. Use a game-side client (our harness, neuro-api client, Tony's testgame HTML, the Unity TicTacToe example). Port 8000 conflicts if both run.
- Do not send speech_finished with isFinal:true until the audio has actually finished (or been cancelled). Games gate turn flow on it ('You should almost always wait for `isFinal` to be `true`').
- Licensing:
- The neuro-sdk docs and SDKs are MIT (Copyright (c) 2024 Vedal AI). If spec text or Randy code is copied into our repo, include the MIT notice. Implementing the protocol from scratch needs no notice.
- Do not reuse neuro-sdk Assets/icon.png or any Neuro-sama likeness.
- Do not name or market the product 'Neuro' or imply VedalAI endorsement. Neutral wording like 'compatible with the Neuro Game SDK protocol' is the safe phrasing.
- Trademark registrations were NOT verified.
- Game mods (LGPL-2.1/AGPL-3.0/MIT) talk to us over a socket, so they impose no license obligations on our server.
- neuro-api (LGPL-3.0) should stay a dev/test dependency only.

## Open questions

- Exact number of automatic force retries in production Neuro. We assume 3 (Gary's choice); make it configurable.
- Does the production server send actions/reregister_all on connect? It is still labelled a proposal ('do not exist yet'), but both official SDKs and Randy implement or send it.
- Does the production server keep a game's action registry across a disconnect (it is 'mostly stateless'), or drop it? We drop it and rely on reregister_all.
- Is speech_finished broadcast to all connected games or only the game that triggered the speech? We broadcast.
- Is held context delivered before or after the pending action's result content? We deliver the result first, then held context in order.
- Force from a different game while one is active: production behaviour is undocumented. We queue it FIFO; same-game forces cancel and replace.
- Does a timed-out result (20 s) during a force trigger a retry like success=false does? The spec says it is 'treated as failed', so we retry.
- Should characterId stay 'pailin', or should some integrations need an alias ('neuro')? None of the six official examples branch on it.
- Phase 2: implement the voice side channel (48 kHz f32 PCM, per-speaker STT attribution, PTT via voice/speaking) for games with voice chat and Discord-like voice-only clients.
- Trademark status of 'Neuro-sama'/'Neuro' was not checked. Get legal review before any public release that mentions compatibility.
