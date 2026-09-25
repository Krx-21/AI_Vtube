# tests/fixtures

Small, licence-clean files that the default (offline, no GPU, no audio) test run uses.

| Path | What | Provenance / licence |
|---|---|---|
| `silero_vad.onnx` | Silero VAD v6.2.3 | MIT, see `silero_vad.LICENSE.txt`; sha256 `1a153a22…8788e3` (same as `models/manifest.toml`) |
| `irc/twitch_sample.txt` | Twitch IRC lines with real tag layouts (PRIVMSG, bits, sub/resub/subgift/raid USERNOTICE, shared chat, a duplicate id); `{channel}` placeholder | Synthetic, written for this project. Kept identical to `aivtube.testing.fakes.SAMPLE_IRC_LINES` |
| `youtube/live_chat_page.json` | A first `liveChatMessages.list` page (the backlog): text messages from a viewer, a moderator and a member, plus a THB super chat | Synthetic, built with `aivtube.testing.fakes.yt_text_message`/`yt_super_chat` |
| `sse/llamacpp_text.sse` | Raw `/v1/chat/completions` stream: Thai text whose combining marks arrive in separate deltas, `finish_reason: length`, final `timings` with `cache_n > 0` | Recorded from a CPU llama-server (commit 1ab7e5a) running Typhoon2.5-Qwen3-4B Q4_K_M with the repo chat template; request in `*.request.json` |
| `sse/llamacpp_tool_call.sse` | Raw stream of one fragmented `set_stream_title` tool call (`{"title": "ไพลินเล่นเกม Minecraft"}`), `finish_reason: tool_calls` | Same server |
| `sse/llamacpp_props.json` / `llamacpp_props_no_template.json` | `/props` excerpts with and without a tool-capable chat template (`supports_tool_calls` true/false) | Same server; the second is the official typhoon-ai GGUF without `--chat-template-file` |

Replay a recording over real HTTP with
`SseFixtureServer(script=[FakeReply("", "", raw_sse=path.read_bytes())])`, or feed the bytes to
`httpx2.MockTransport`. Never commit model weights or voice recordings here.
