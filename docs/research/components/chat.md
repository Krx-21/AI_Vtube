# Live chat platforms — component brief

_Research snapshot: 2026-09-25. Verified facts carry a source; unverified items are marked._

## Recommendation

Build a small `chat/` package of our own. Each platform gets an adapter behind two protocols: `ChatSource` (an async iterator of normalised `ChatMessage`) and `ChannelActions` (`send`, `timeout`, `poll`, `set_title`). Each protocol also gets an in-memory fake, so everything can be unit-tested in the Linux container with no GPU and no network. Twitch and YouTube use only httpx + websockets (+ grpcio for YouTube streaming). TikTok is an optional plugin that stays off by default.

TWITCH (main platform):
- Reading: raw EventSub WebSocket at wss://eventsub.wss.twitch.tv/ws. Subscribe to `channel.chat.message` v1 and `channel.chat.notification` v1. These two cover subs, resubs, gift subs, raids and watch streaks with only `user:read:chat`. Bits come from `event.cheer.bits`. Optionally add `channel.channel_points_custom_reward_redemption.add` (Neuro has redeems).
- Actions: Helix over httpx.
- Auth: ONE broadcaster user token from the Device Code Grant Flow, with the app registered as a Public client. That means no client secret and no local HTTP callback server. Polls and title changes require `broadcaster_id` to equal the token's user. Timeouts need a moderator, and the broadcaster counts as one. So a single broadcaster token covers everything.
- Scopes: `user:read:chat user:write:chat moderator:manage:banned_users channel:manage:polls channel:manage:broadcast` (+ `channel:read:redemptions bits:read` if needed).
- Chat identity: the AI mainly speaks through TTS, so chat sending is optional. If Pailin should post as her own account, add a second DCF token for a bot account (`user:write:chat` only) and make that account a mod or VIP. That raises its limit from 20 to 100 messages per 30 s.
- Keep anonymous IRC (`justinfan` on wss://irc-ws.chat.twitch.tv:443) as a zero-config read-only mode for development and demos, and as a fallback. It is verified working today.
- Do NOT adopt twitchio 3.3.2 as the core. It is a full bot framework (commands/components, aiohttp, its own token store, IRC removed) that would duplicate our serial orchestrator. The raw path is about 150 LOC, easier to fake, and has no framework lock-in. twitchio is a valid alternative if the team wants AutoBot/Conduits later.

YOUTUBE:
- Reading needs only an API key. Primary: `liveChatMessages.streamList` over gRPC (youtube.googleapis.com:443, `x-goog-api-key` metadata). The official guide says an API key or OAuth works, and it is push-based with the lowest latency. Fallback: `liveChatMessages.list` polling that honours `pollingIntervalMillis`. It costs 1 unit per call per the quota table updated 2026-09-15.
- Discovery, in order: (a) the owner's `liveBroadcasts.list?broadcastStatus=active` with OAuth (1 unit); (b) a configured video id or the scraped `https://www.youtube.com/@handle/live`, then `videos.list?part=liveStreamingDetails` to get `activeLiveChatId` (1 unit, API key); (c) `search.list eventType=live`, which since 2026-06-01 has its own bucket of 100 calls/day.
- Actions (optional) need owner OAuth with scope `youtube.force-ssl`, from google-auth-oauthlib InstalledAppFlow. Each action costs 50 units against a 10,000/day budget, so rate-limit them hard.
- Publish the Google OAuth consent screen out of "Testing". Otherwise refresh tokens expire after 7 days.
- Never use pytchat (archived).

TIKTOK:
- There is no official read API. TikTokLive==7.0.1 (2026-09-10) is a reverse-engineered, read-only client that depends on the Euler Stream sign server. Its authors call it "not production-ready".
- Run it as an optional adapter with a pinned version, isolated so it cannot crash the core. Treat it as best-effort.
- Sending chat through Euler needs a paid "premium" route plus an OAuth session, so treat TikTok as read-only.
- License: a modified AGPL-3.0. The §18 exception explicitly covers "TikTok LIVE Stream Bots" as long as nothing is offered as hosted SaaS (§19).

SELECTOR: Neuro's per-turn chat count and priorities are publicly unknown (corpus C11: "do not guess"). Our design is therefore our own, inspired by "picks what to respond to within a limited window" (T1):
- Keep a rolling buffer of about 40 s. Donations, subs and raids go to a separate must-acknowledge queue with a 10-minute TTL.
- At each serial decision tick, take k=3 messages by softmax-sampling over a score. The score combines recency decay, a mention of the character name, question shape, sub/mod/VIP/first-message flags, a copypasta penalty, and a per-user cooldown.
- Keep at most one message per user and one per near-duplicate.
- Drop Twitch shared-chat messages that come from other channels.
- Consume the window after each decision. Chat that arrives mid-decision is queued and merged into the next tick, matching Neuro's serial loop (T1).

INPUT SAFETY (before the LLM):
- Tier 0, synchronous: NFKC, strip zero-width characters (U+200B is common in Thai), collapse repeated characters, then an Aho-Corasick/regex blocklist (Thai + English) and PII/URL regexes.
- Tier 1, asynchronous: run `typhoon-ai/typhoon2-safety-preview` only on the few selected candidates and display names, not on all chat. It is an MIT-licensed mDeBERTa-v3-base binary classifier trained on Thai sensitive topics, including The Monarchy, which matters for Thai law (§112). Qwen3Guard-Gen-0.6B is the multilingual option with a Jailbreak category.
- Tier 2: a human moderator panel with a kill switch, since Vedal: "there currently needs to be a human there to moderate" (T1).
- Always pass chat to the LLM as quoted, delimited, untrusted user data, never in the system role.

## Alternatives

### twitchio 3.3.2 (framework) instead of raw httpx+websockets
- **Pros:** Maintained (released 2026-08-04). Typed models, EventSub WS/webhook/Conduits, built-in DCF + token store + OAuth web adapter, commands/components, routines.
- **Cons:** Python >=3.11, pulls aiohttp. Opinionated Bot lifecycle that duplicates our orchestrator. IRC removed (no anonymous mode). Harder to fake in unit tests. Framework upgrades can break us.
- **When:** If we later add many Twitch features (redeems, conduits, multiple channels) and accept the framework, or if the team prefers not to own ~150 LOC of protocol code.

### twitchAPI (pyTwitchAPI) 4.5.0
- **Pros:** Covers Helix + EventSub + Chat. MIT license.
- **Cons:** Last release 2025-05-23, so it may lag 2026 changes (pin, gif fragments).
- **When:** Only if twitchio is rejected and the team still wants a library.

### Twitch IRC only (authenticated, chat:read/chat:edit)
- **Pros:** Simplest protocol, one socket for read + send, anonymous read mode, no Helix needed for chat.
- **Cons:** Twitch calls it limited and harder to parse. No structured notifications (use USERNOTICE tags). Moderation/polls/title still need Helix. Future deprecation risk.
- **When:** Dev/demo mode and a read-only fallback when tokens are missing or expired.

### YouTube list polling only
- **Pros:** Plain REST via httpx or google-api-python-client 2.200.0. Documented quota (1 unit/call). Trivial to fake.
- **Cons:** Latency up to pollingIntervalMillis. Quota burn over long streams.
- **When:** Fallback when gRPC fails, or for the first milestone.

### YouTube streamList via REST (/youtube/v3/liveChat/messages/stream)
- **Pros:** No grpcio dependency, httpx only.
- **Cons:** Streamed JSON-array framing needs an incremental parser. Not in the official guide. Not verified with a real key.
- **When:** If grpcio wheels are a problem (currently available for cp310-cp315 win_amd64).

### YouTube via InnerTube scraping (pytchat/chat-downloader style)
- **Pros:** No API key or quota.
- **Cons:** pytchat archived, chat-downloader stale. Breaks with YouTube changes. ToS grey area. Bot-checks from some IPs.
- **When:** Avoid, except as a last-resort read-only fallback maintained by us.

### TikTok via Euler Stream hosted WebSocket API (paid)
- **Pros:** Euler maintains the reverse-engineering. More stable than self-signing. Higher limits. Premium chat send.
- **Cons:** $50/mo Business tier. Third-party dependency and data relay. Still unofficial.
- **When:** Only if the streamer actually streams on TikTok regularly and needs reliability.

### Qwen3Guard-Gen-0.6B instead of typhoon2-safety-preview for tier 1
- **Pros:** Apache-2.0, 119 languages, 3-level severity plus categories incl. Jailbreak/PII, and a streaming variant that can also check LLM output.
- **Cons:** Generative, so slower and heavier (1.5 GB). Not specialised for Thai sensitive topics (monarchy etc.).
- **When:** As an output-side (response) guard, or combined with Typhoon for jailbreak detection.

## Verified facts

- ✅ Twitch IRC is NOT deprecated as of Sept 2026. The docs say EventSub + API is 'preferred' / 'recommended'. Only non-secure WebSocket connections were decommissioned (2025-08-15). The WebSocket non-SSL column reads 'Unavailable (see announcement)'. wss://irc-ws.chat.twitch.tv:443 and irc://irc.chat.twitch.tv:6697 remain. IRC docs got new tags in 2026-06-18 and 2026-07-17 (gif tag).  
  Source: https://dev.twitch.tv/docs/chat/irc/ ; https://dev.twitch.tv/docs/rss/change-log.xml (entries 2025-08-22, 2026-06-18, 2026-07-17) ; https://discuss.dev.twitch.com/t/decommission-of-non-secure-websocket-connections-to-twitch-irc-servers/64142
- ✅ Anonymous read-only IRC still works. Login was CAP REQ twitch.tv/tags twitch.tv/commands, PASS SCHMOOPIIE, NICK justinfan<5 digits>, JOIN #xqc etc. The server returned 001 Welcome, ROOMSTATE and live PRIVMSG with full tags (badge-info, badges, id, user-id, tmi-sent-ts, first-msg, returning-chatter, mod, subscriber). justinfan is undocumented, but a live test from this container on 2026-09-25 succeeded.  
  Source: live test: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/chat/irc_anon.py
- ✅ Authenticated IRC needs a user token with scopes chat:read (read) and chat:edit (send): 'PASS oauth:<token>', 'NICK <login>'.  
  Source: https://dev.twitch.tv/docs/chat/irc/
- ✅ channel.chat.message v1 authorization: 'Requires user:read:chat scope from the chatting user. If app access token used, then additionally requires user:bot scope from chatting user, and either channel:bot scope from broadcaster or moderator status.' Condition is {broadcaster_user_id, user_id}. The event carries chatter_user_id/name, message_id, message.text + fragments (text|cheermote|emote|mention|gif), badges[{set_id,id,info}], cheer{bits}, reply, channel_points_custom_reward_id, source_broadcaster_user_id/source_message_id/is_source_only (shared chat).  
  Source: https://dev.twitch.tv/docs/eventsub/eventsub-subscription-types/ ; https://dev.twitch.tv/docs/eventsub/eventsub-reference/
- ✅ channel.chat.notification notice_type values: sub, resub, sub_gift, community_sub_gift, gift_paid_upgrade, prime_paid_upgrade, raid, unraid, pay_it_forward, announcement, bits_badge_tier, charity_donation, watch_streak, modiversary, plus shared_chat_* variants and unknown. Same auth as chat.message (user:read:chat).  
  Source: https://dev.twitch.tv/docs/eventsub/eventsub-reference/
- ✅ EventSub WebSocket: URL wss://eventsub.wss.twitch.tv/ws?keepalive_timeout_seconds=10..600. You must subscribe within 10 s of session_welcome (close 4003). WebSockets require user access tokens: 'If you use app access tokens with WebSockets, the subscriptions will fail.' Limits per (client_id, user_id): max 3 connections, 300 enabled subs per connection, max_total_cost 10. On session_reconnect you have 30 s to connect to reconnect_url (close 4004). A live connect returned session_welcome and then session_keepalive after 10 s.  
  Source: https://dev.twitch.tv/docs/eventsub/handling-websocket-events/ ; live test es_welcome.py 2026-09-25
- ✅ Send Chat Message: POST https://api.twitch.tv/helix/chat/messages. A user token needs user:write:chat (an app token needs user:write:chat + user:bot, plus channel:bot or mod). Body: broadcaster_id, sender_id (must match token user), message (max 500 chars), reply_parent_message_id, for_source_only (app token only; with a user token it returns 400), pin (NEW 2026-05-15, needs moderator:manage:chat_messages). Response: data[0].message_id, is_sent, drop_reason{code,message}. Errors 422 too large, 429 rate-limited.  
  Source: https://dev.twitch.tv/docs/api/reference/#send-chat-message
- ✅ Chat send limits: 20 msgs/30 s if the sender is not broadcaster/mod/VIP, and 100/30 s if it is. Non-mod: max 1 msg/s per channel. Exceeding the limit means 'Twitch ignores the bot messages for 1 hour, except in chats where the chatbot's user account is the broadcaster, a moderator, or a VIP.' Verified bots get 7500/30 s, but chatbot verification reviews are 'temporarily paused'. The Chat Bot badge requires an app access token + channel:bot and a non-broadcaster account.  
  Source: https://dev.twitch.tv/docs/chat/
- ✅ Timeout: POST https://api.twitch.tv/helix/moderation/bans?broadcaster_id=&moderator_id= with body {"data":{"user_id":"...","duration":1..1209600,"reason":"<=500 chars"}}. Omitting duration means a permanent ban. Scope moderator:manage:banned_users (an app token needs +user:bot).  
  Source: https://dev.twitch.tv/docs/api/reference/#ban-user
- ✅ Create Poll: POST https://api.twitch.tv/helix/polls. Needs a user token with channel:manage:polls, and broadcaster_id must match the token user. Title max 60 chars, 2-5 choices each max 25 chars, duration 15-1800 s, one poll at a time, choice titles go through AutoMod.  
  Source: https://dev.twitch.tv/docs/api/reference/#create-poll
- ⚠️ unverified — Polls are Affiliate/Partner only. The API returns 403 'ownedBy <id> is not a partner or affiliate'. The official API reference does not state this; the source is the dev forum and help articles.  
  Source: https://discuss.dev.twitch.com/t/do-i-need-to-be-an-affiliate-to-create-a-poll/41147
- ✅ Modify channel title: PATCH https://api.twitch.tv/helix/channels?broadcaster_id= with body {"title":"..."} (must not be empty). Scope channel:manage:broadcast; broadcaster_id must match the token user. Returns 204.  
  Source: https://dev.twitch.tv/docs/api/reference/#modify-channel-information
- ✅ Device Code Grant Flow (GA since 2023-12-05): POST https://id.twitch.tv/oauth2/device (form client_id, scopes) returns device_code, expires_in (1800), interval (5), user_code, verification_uri (https://www.twitch.tv/activate?public=true&device-code=...). Then poll POST https://id.twitch.tv/oauth2/token with client_id, scopes, device_code, grant_type=urn:ietf:params:oauth:grant-type:device_code; the pending response is {status:400,message:'authorization_pending'}. Public clients need no secret. The docs suggest public clients for Windows. Access tokens last about 4 h. DCF refresh tokens are one-time use and expire after 30 days of inactivity. Endpoints answered as expected in a live probe (invalid client gives 400 'invalid client').  
  Source: https://dev.twitch.tv/docs/authentication/getting-tokens-oauth/#device-code-grant-flow
- ✅ Refresh: POST https://id.twitch.tv/oauth2/token with grant_type=refresh_token, refresh_token (URL-encoded), client_id (client_secret omitted for public clients). Apps must call GET https://id.twitch.tv/oauth2/validate (header 'Authorization: OAuth <token>') at startup and hourly; Twitch audits this.  
  Source: https://dev.twitch.tv/docs/authentication/refresh-tokens/ ; https://dev.twitch.tv/docs/authentication/validate-tokens/
- ✅ App registration requires 2FA on the Twitch account and at least one OAuth Redirect URL in the console form, even if you only use DCF.  
  Source: https://dev.twitch.tv/docs/authentication/register-app/
- ✅ Helix rate-limit bucket: headers Ratelimit-Limit (example 800) / Ratelimit-Remaining / Ratelimit-Reset. Send Chat Message uses a separate bucket.  
  Source: https://dev.twitch.tv/docs/api/guide/ ; https://dev.twitch.tv/docs/chat/
- ✅ twitchio 3.3.2 (PyPI 2026-08-04): requires_python >=3.11, depends on aiohttp<4,>=3.9.1. 'IRC was removed from the core of TwitchIO... chat... is now done via EventSub'. Supports EventSub over webhook/websocket/conduits and has built-in Device Code Flow (OAuth.device_code_flow / device_code_authorization). twitchAPI (pyTwitchAPI) 4.5.0 was last released 2025-05-23.  
  Source: https://pypi.org/pypi/twitchio/json ; https://twitchio.dev/en/latest/getting-started/migrating.html ; twitchio-3.3.2 wheel source twitchio/authentication/oauth.py
- ✅ YouTube liveChatMessages.list: GET https://www.googleapis.com/youtube/v3/liveChat/messages with required liveChatId and part (id,snippet,authorDetails), plus maxResults 200-2000 (default 500), pageToken, profileImageSize 16-720, hl. The response has pollingIntervalMillis, nextPageToken, offlineAt, activePollItem. The docs now point pollers to streamList 'to avoid exceeding your quota'.  
  Source: https://developers.google.com/youtube/v3/live/docs/liveChatMessages/list
- ✅ YouTube quota table (last updated 2026-09-15): liveChatMessages.list = 1, insert = 50, delete = 50, transition = 50; liveChatBans insert/delete = 50; liveBroadcasts.list = 1; videos.list = 1. Default 10,000 units/day, reset at midnight PT. Since 2026-06-01, search.list and videos.insert have their own buckets (100 calls/day, 1 per call).  
  Source: https://developers.google.com/youtube/v3/determine_quota_cost ; https://developers.google.com/youtube/v3/revision_history (2026-06-01)
- ✅ liveChatMessages.streamList exists (docs updated 2025-07-14). It is a gRPC server-stream: service youtube.api.v3.V3DataLiveChatMessageService/StreamList at dns:///youtube.googleapis.com:443, with stream_list.proto (proto2) published in the guide. Auth: 'You can use an OAuth 2.0 access token or an API key' via metadata ('x-goog-api-key', KEY) or ('authorization', 'Bearer ' + token). The first response carries recent history, then pushes. Resume with pageToken=nextPageToken. RESOURCE_EXHAUSTED if requests come too fast.  
  Source: https://developers.google.com/youtube/v3/live/streaming-live-chat ; https://developers.google.com/youtube/v3/live/docs/liveChatMessages/streamList
- ✅ A REST mapping of streamList exists: GET https://youtube.googleapis.com/youtube/v3/liveChat/messages/stream, present in discovery doc revision 20260924 as youtube.v3.liveChat.messages.stream. It returns a streamed JSON array; alt=sse is rejected with 400. The path answered 403 (not 404) without credentials; it was not exercised with a real key.  
  Source: https://youtube.googleapis.com/$discovery/rest?version=v3 ; live curl probes 2026-09-25
- ⚠️ unverified — Anonymous calls to liveChat/messages fail with 403 'Method doesn't allow unregistered callers... Please use API Key or other form of API consumer identity', which implies an API key is accepted. The list docs do not explicitly state API-key support for public chats; the streamList guide does.  
  Source: live curl probe 2026-09-25 ; https://developers.google.com/youtube/v3/live/streaming-live-chat
- ✅ The liveChatId comes from videos.list part=liveStreamingDetails -> liveStreamingDetails.activeLiveChatId, or from liveBroadcasts.list (OAuth; broadcastStatus=active|all|upcoming|completed, broadcastType=all|event|persistent) -> snippet.liveChatId.  
  Source: https://developers.google.com/youtube/v3/live/streaming-live-chat ; discovery doc
- ✅ Scraping https://www.youtube.com/@handle/live returns a watch page whose ytInitialData.currentVideoEndpoint.watchEndpoint.videoId is the live video, with '"isLive":true' present. From a datacenter IP the player response says 'Sign in to confirm you're not a bot', but ytInitialData still carries the id. Channels with several concurrent streams return varying ids. This is unofficial and fragile.  
  Source: live test youtube_raw.live_video_id_from_handle on @LofiGirl/@NASA/@YouTube 2026-09-25
- ✅ YouTube message types: textMessageEvent, superChatEvent (amountMicros, currency, amountDisplayString, userComment, tier), superStickerEvent, newSponsorEvent, memberMilestoneChatEvent, membershipGiftingEvent, giftMembershipReceivedEvent, userBannedEvent, pollEvent, giftEvent (NEW 2026-03-26: giftName, jewelsAmount, comboCount; the same id is reused to update the combo count), tombstone, chatEndedEvent. messageDeleted/RetractedEvent were removed from the docs 2026-06-23. authorDetails: channelId, displayName, isChatOwner, isChatModerator, isChatSponsor (member), isVerified.  
  Source: discovery doc schemas ; https://developers.google.com/youtube/v3/live/revision_history
- ✅ YouTube actions: liveChatMessages.insert (scope youtube or youtube.force-ssl) with snippet.type=textMessageEvent + textMessageDetails.messageText, or type=pollEvent + pollDetails.metadata.questionText + options[2..4].optionText (error 'A pinned active poll already exists'). Close a poll with liveChatMessages.transition?id=&status=closed. liveChatBans.insert snippet{liveChatId, type: permanent|temporary, banDurationSeconds, bannedUserDetails.channelId} must be authorized by the owner or a moderator and cannot ban the owner or mods.  
  Source: https://developers.google.com/youtube/v3/live/docs/liveChatMessages/insert ; .../liveChatBans/insert ; .../liveChatMessages/transition
- ✅ Google OAuth: an External consent screen in 'Testing' status issues refresh tokens that expire in 7 days, unless the only scopes are name/email/profile. Limit: 100 refresh tokens per Google Account per client id.  
  Source: https://developers.google.com/identity/protocols/oauth2
- ✅ pytchat 0.5.5 was last released 2021-07-24; the repo was archived by its owner on 2022-01-25. chat-downloader 0.2.8 was last released 2023-09-03 and is stale; whether it works in 2026 was not tested.  
  Source: https://pypi.org/pypi/pytchat/json ; https://github.com/taizan-hokuto/pytchat ; https://pypi.org/pypi/chat-downloader/json
- ✅ TikTokLive 7.0.1 (PyPI 2026-09-10; 7.0.0 on 2026-08-18): requires_python >=3.10. Deps: betterproto2==0.9.1, websockets_proxy==0.1.3, httpx>=0.28.1, TikTokLiveProto==0.2.2, EulerApiSdk==0.1.0, pyee==13.0.1. Default tiktok_sign_url='https://api.eulerstream.com', webcast URL https://webcast.tiktok.com/webcast. README: 'This is not a production-ready API. It is a reverse engineering project.' CommentEvent.content (alias .comment), user.is_moderator, user_is_super_fan. GiftEvent has repeat_count/repeat_end/streaking and value = repeat_count*diamond_count*0.005 USD.  
  Source: https://pypi.org/pypi/TikTokLive/json ; https://github.com/isaackogan/TikTokLive ; wheel source TikTokLive/client/web/web_settings.py, events/proto_events.py
- ✅ TikTokLive license is AGPL-3.0 with §7 additions. §18 lets you integrate into downstream apps without AGPL, and names 'TikTok LIVE Stream Bots'. §19 revokes that for commercial, closed-source or hosted SaaS, WebSocket relays or data APIs. §21 excepts TikFinity and Euler Stream.  
  Source: https://raw.githubusercontent.com/isaackogan/TikTokLive/master/LICENSE
- ⚠️ unverified — Euler Stream Community tier is $0 with 2,500 requests/day and 25 cloud WebSockets; Business is $50/mo with 10,000/day. Exact keyless (no API key) limits are unclear.  
  Source: https://www.eulerstream.com/pricing (via WebFetch summary)
- ✅ An open issue (2026-09-10, TikTokLive v7.0.0, community key) reports WebSocket URIs pointing at wss://ws-fallback.eulerstream.com never reaching ConnectEvent and 400 at handshake, i.e. traffic can be relayed through Euler infrastructure. Its resolution is unknown.  
  Source: https://github.com/EulerStream/TikTok-Live-Api/issues/9
- ✅ No official TikTok API exists for reading LIVE chat. Sending chat via Euler needs the premium route send_room_chat with an OAuth session.  
  Source: TikTokLive README FAQ ; EulerApiSdk-0.1.0 api/tik_tok_live_premium/send_room_chat.py
- ✅ typhoon-ai/typhoon2-safety-preview: an MIT-licensed binary classifier (0=Unharm, 1=Harmful) fine-tuned from microsoft/mdeberta-v3-base, trained on Thai Sensitive Topics (incl. The Monarchy, Political Divide, Gambling, Vape, Cannabis...) + WildGuard. model.safetensors is 1,115,268,200 bytes (fp32). Its card reports a Thai-content average of 76.1 vs LlamaGuard3-8B at 56.7. The card's code still says scb10x/...; use the typhoon-ai/ path.  
  Source: https://huggingface.co/typhoon-ai/typhoon2-safety-preview
- ✅ Qwen/Qwen3Guard-Gen-0.6B (Apache-2.0, safetensors 1.5 GB) supports 119 languages and outputs 'Safety: Safe|Unsafe|Controversial' plus categories incl. Violent, PII, Politically Sensitive Topics, Jailbreak. Qwen3Guard-Stream-0.6B does token-level streaming moderation.  
  Source: https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
- ✅ Neuro facts relevant here, from the corpus: T1 'picks what to respond to within a limited window'; serial loop that merges context arriving mid-decision (T1); human moderation plus AI filters (T1, Vice 2023-01-04); input-side chat filtering since the 2021 Airis prototype (T5); per-platform extra filter on Bilibili (T4). Per-turn message count and sub prioritisation are UNKNOWN; do not guess.  
  Source: /root/.claude/uploads/02efebb2-c032-5a2c-81bb-f71e8a5461a3/ed3f3f77-C11-safety.md
- ✅ Current library versions (PyPI, Sept 2026): websockets 17.1 (py>=3.11), httpx 0.28.1, aiohttp 3.14.3, grpcio / grpcio-tools 1.84.0 (win_amd64 wheels for cp310-cp315), protobuf 7.36.2, google-api-python-client 2.200.0, google-auth-oauthlib 1.4.1, pythainlp 5.3.8, pyahocorasick 2.3.1, transformers 5.17.0, ijson 3.5.1.  
  Source: https://pypi.org/pypi/<name>/json
- ✅ The prototype ChatMessage model, IRC/EventSub/YouTube parsers and ChatWindow selector pass the scratch test run on real captured IRC lines and the doc payload examples. The raw Twitch and YouTube clients compile, and the anonymous IRC + @handle/live helpers were exercised live.  
  Source: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/chat/proto/{chatmodel.py,test_chatmodel.py,twitch_raw.py,youtube_raw.py}

## Install (Windows)

```
Windows 11, Python 3.12 (3.11 minimum for websockets 17 / twitchio 3), in PowerShell:

  py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
  pip install "httpx==0.28.1" "websockets>=15,<18"          # Twitch EventSub/IRC/Helix, YouTube REST
  pip install "grpcio==1.84.0" "protobuf>=5"                  # YouTube streamList (runtime)
  pip install "grpcio-tools==1.84.0"                          # dev only: regenerate stubs
  pip install "google-auth-oauthlib==1.4.1"                   # only if YouTube actions (owner OAuth) are enabled
  pip install "pyahocorasick==2.3.1" "pythainlp==5.3.8"       # tier-0 filter (pythainlp optional: Thai normalize/tokenize)
  pip install "transformers>=4.46" torch                      # tier-1 classifier (use the CUDA torch wheel index the LLM/STT components already choose)
  pip install "TikTokLive==7.0.1"                             # OPTIONAL extra, e.g. pyproject [project.optional-dependencies] tiktok = ["TikTokLive==7.0.1"]

Stub generation, done once and the output committed so Windows users never need protoc: save stream_list.proto from https://developers.google.com/youtube/v3/live/streaming-live-chat into src/.../youtube/proto/, then run
  python -m grpc_tools.protoc -I src/pkg/youtube/proto --python_out=src/pkg/youtube/proto --grpc_python_out=src/pkg/youtube/proto stream_list.proto
The generated stubs use absolute `import stream_list_pb2`, so fix that to a relative import after generation, or put the folder on sys.path.

Twitch app, one-time:
1. Enable 2FA on the Twitch account.
2. Go to https://dev.twitch.tv/console/apps -> Register Your Application. Set Name "pailin-vtuber-<user>" (it must be unique), OAuth Redirect URL http://localhost:3000 (required by the form; unused by DCF), Category "Chat Bot", Client Type = Public.
3. Copy the Client ID into config. There is no secret with a Public client.
4. On first run the app prints the verification_uri + user_code. The streamer opens it while logged in as the broadcaster. Store the tokens in %APPDATA%\\ai-vtube\\twitch_tokens.json and keep the file out of git.

YouTube, one-time:
1. Google Cloud console: create a project and enable "YouTube Data API v3".
2. Create an API key and restrict it to the YouTube Data API v3. That is enough for reading.
3. For actions: OAuth consent screen External -> add the youtube.force-ssl scope -> create an OAuth client of type "Desktop app" -> download client_secret.json. Move the consent screen to "In production"; otherwise refresh tokens die after 7 days. Unverified-app warnings are acceptable for personal use (UNVERIFIED policy detail).
4. First run: google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file('client_secret.json', ['https://www.googleapis.com/auth/youtube.force-ssl']).run_local_server(port=0).

Firewall: all connections are outbound (443). No inbound ports are needed, which is why EventSub WebSocket is used rather than webhooks.

CI (ubuntu + windows): no network is needed. Adapters take injected transport factories, and tests use the fakes plus recorded fixtures (IRC lines, EventSub JSON from the docs, YouTube list JSON).
```

## API notes

=== TWITCH ===
Auth (Device Code Flow, public client):
  POST https://id.twitch.tv/oauth2/device  form: client_id, scopes="user:read:chat user:write:chat moderator:manage:banned_users channel:manage:polls channel:manage:broadcast"
   -> {"device_code","expires_in":1800,"interval":5,"user_code","verification_uri":"https://www.twitch.tv/activate?public=true&device-code=..."}
  POST https://id.twitch.tv/oauth2/token  form: client_id, scopes, device_code, grant_type=urn:ietf:params:oauth:grant-type:device_code
   -> 400 {"status":400,"message":"authorization_pending"} until approved, then {"access_token","expires_in","refresh_token","scope":[...],"token_type":"bearer"}
  Refresh: POST https://id.twitch.tv/oauth2/token  form: client_id, grant_type=refresh_token, refresh_token(urlencoded). The refresh token is ONE-TIME: persist the new pair atomically.
  Validate: GET https://id.twitch.tv/oauth2/validate  header "Authorization: OAuth <token>" -> {client_id, login, scopes, user_id, expires_in}. Call at start and every hour; on 401 refresh.
  Helix headers: "Authorization: Bearer <token>", "Client-Id: <client_id>". Resolve ids: GET https://api.twitch.tv/helix/users?login=<name>.

Read (EventSub WebSocket):
  connect wss://eventsub.wss.twitch.tv/ws?keepalive_timeout_seconds=30
  <- {"metadata":{"message_type":"session_welcome",...},"payload":{"session":{"id":"AgoQ...","keepalive_timeout_seconds":30,"reconnect_url":null}}}
  Within 10 s: POST https://api.twitch.tv/helix/eventsub/subscriptions
     {"type":"channel.chat.message","version":"1","condition":{"broadcaster_user_id":"<B>","user_id":"<U reading user>"},"transport":{"method":"websocket","session_id":"<id>"}}
     Repeat for "channel.chat.notification" v1 (same condition) and optionally "channel.channel_points_custom_reward_redemption.add" v1 {"broadcaster_user_id"} (scope channel:read:redemptions).
  <- notification: payload.subscription.type + payload.event. Dedupe on metadata.message_id and event.message_id.
  <- session_keepalive (no event within the keepalive window means reconnect); session_reconnect: connect to payload.session.reconnect_url within 30 s, subs carry over, do NOT resubscribe; revocation.
  Close codes: 4001 client sent traffic, 4002 ping-pong fail, 4003 unused (no sub within 10 s), 4004 reconnect grace expired, 4005/4006 network, 4007 invalid reconnect.
  Shared chat: when event.source_broadcaster_user_id is set and != broadcaster, the message is from another channel; skip it or down-weight it.

Read (anonymous IRC fallback, verified live):
  wss://irc-ws.chat.twitch.tv:443 -> "CAP REQ :twitch.tv/tags twitch.tv/commands", "PASS SCHMOOPIIE", "NICK justinfan12345", "JOIN #channel"; answer "PING :tmi.twitch.tv" with "PONG :tmi.twitch.tv".
  Tags: badges=subscriber/2018,moderator/1,vip/1,broadcaster/1; badge-info=subscriber/19; bits=<n> on cheers; id; user-id; display-name; tmi-sent-ts (ms); first-msg; mod; subscriber; source-room-id (shared chat). The server may resend a message, so dedupe on id.

Actions (Helix):
  POST /helix/chat/messages  {"broadcaster_id":B,"sender_id":U,"message":"<=500","reply_parent_message_id":opt} -> data[0].{message_id,is_sent,drop_reason}
  POST /helix/moderation/bans?broadcaster_id=B&moderator_id=U  {"data":{"user_id":X,"duration":60,"reason":"..."}}   (duration 1..1209600; omit = permanent ban; use AI-initiated timeouts only, never AI bans)
  POST /helix/polls  {"broadcaster_id":B,"title":"<=60","choices":[{"title":"<=25"}x2..5],"duration":15..1800}   (Affiliate/Partner; one at a time)
  PATCH /helix/channels?broadcaster_id=B  {"title":"..."} -> 204
Full runnable raw client (DCF, validate, refresh, EventSub loop with reconnect, Helix actions, anonymous IRC): /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/chat/proto/twitch_raw.py

Core of the EventSub loop:
  async with websockets.connect(url) as ws:
      while True:
          msg = json.loads(await asyncio.wait_for(ws.recv(), keepalive + 10))
          mt = msg["metadata"]["message_type"]
          if mt == "session_welcome" and first: [await helix.subscribe(sid, t, c) for t, c in subs]
          elif mt == "notification": await on_event(msg["payload"]["subscription"]["type"], msg["payload"]["event"])
          elif mt == "session_reconnect": url = msg["payload"]["session"]["reconnect_url"]; break

twitchio 3.3.2 equivalent: subclass twitchio.ext.commands.Bot/AutoBot with client_id, bot_id, owner_id. Tokens come from add_token()/its OAuth adapter at http://localhost:4343/oauth or its device_code_flow helpers, and you subscribe with twitchio.eventsub.ChatMessageSubscription(broadcaster_user_id=..., user_id=...). It is not recommended for the core.

=== YOUTUBE ===
Discovery (pick the first that works):
  OAuth owner: GET https://www.googleapis.com/youtube/v3/liveBroadcasts?part=snippet&broadcastStatus=active&broadcastType=all (Bearer) -> items[0].snippet.liveChatId   [1 unit]
  API key:     GET https://www.googleapis.com/youtube/v3/videos?part=liveStreamingDetails&id=<VIDEO>&key=<KEY> -> items[0].liveStreamingDetails.activeLiveChatId   [1 unit]
  Video id without quota (unofficial): GET https://www.youtube.com/@handle/live -> ytInitialData.currentVideoEndpoint.watchEndpoint.videoId, only if '"isLive":true' is present.
  search.list?part=id&channelId=UC..&eventType=live&type=video&key=  (own bucket: 100 calls/day)
Read, polling fallback:
  GET https://www.googleapis.com/youtube/v3/liveChat/messages?liveChatId=<ID>&part=id,snippet,authorDetails&maxResults=2000[&pageToken=<next>]&key=<KEY>
  -> {"pollingIntervalMillis":N,"nextPageToken":"...","offlineAt":?, "items":[liveChatMessage]}. Sleep max(N/1000, 2) s, and skip the backlog on the first page. Cost 1 unit/call: at 5 s that is 720 units/h, about 13 h/day on 10k.
Read, streaming (preferred; official demo pattern):
  channel = grpc.secure_channel("dns:///youtube.googleapis.com:443", grpc.ssl_channel_credentials())
  stub = stream_list_pb2_grpc.V3DataLiveChatMessageServiceStub(channel)
  req = stream_list_pb2.LiveChatMessageListRequest(part=["snippet","authorDetails"], live_chat_id=ID, max_results=20, page_token=next_token)
  for resp in stub.StreamList(req, metadata=(("x-goog-api-key", KEY),)): handle(resp.items); next_token = resp.next_page_token
  Reconnect with page_token=last next_page_token. Proto field names are snake_case; enum snippet.type e.g. TEXT_MESSAGE_EVENT=1, SUPER_CHAT_EVENT=15, GIFT_EVENT=21. Use grpc.aio for asyncio, or run the sync iterator in a thread and push into an asyncio.Queue via loop.call_soon_threadsafe.
  REST alternative: GET https://youtube.googleapis.com/youtube/v3/liveChat/messages/stream?liveChatId=&part=snippet,authorDetails&key= returns a streamed JSON array (use httpx stream + ijson.items_coro or a bracket-depth splitter). The framing was not verified with a real key.
Actions (OAuth Bearer, scope https://www.googleapis.com/auth/youtube.force-ssl, 50 units each):
  POST https://www.googleapis.com/youtube/v3/liveChat/messages?part=snippet  {"snippet":{"liveChatId":ID,"type":"textMessageEvent","textMessageDetails":{"messageText":"..."}}}
  POST .../liveChat/messages?part=snippet  {"snippet":{"liveChatId":ID,"type":"pollEvent","pollDetails":{"metadata":{"questionText":"...","options":[{"optionText":"A"},{"optionText":"B"}]}}}}  (2-4 options)
  POST .../liveChat/messages/transition?id=<pollMsgId>&status=closed&part=snippet
  POST https://www.googleapis.com/youtube/v3/liveChat/bans?part=snippet  {"snippet":{"liveChatId":ID,"type":"temporary","banDurationSeconds":"60","bannedUserDetails":{"channelId":"UC..."}}}
  Title: PUT https://www.googleapis.com/youtube/v3/videos?part=snippet {"id":VIDEO,"snippet":{"title":"...","categoryId":"20"}}. The categoryId must be resent. Not tested.
Budget: 10,000/day means at most ~190 actions/day after reads. Cap AI YouTube actions to about 1 per 2 min.
Runnable helpers: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/chat/proto/youtube_raw.py

=== TIKTOK (optional, read-only) ===
  from TikTokLive import TikTokLiveClient
  from TikTokLive.events import CommentEvent, GiftEvent, ConnectEvent, DisconnectEvent
  client = TikTokLiveClient(unique_id="@streamer")        # optional: client.web.set_... / tiktok_sign_api_key via WebDefaults
  @client.on(CommentEvent)
  async def on_comment(ev): push(ChatMessage(Platform.TIKTOK, str(ev.common.msg_id if ev.common else ""), ChatUser(Platform.TIKTOK, str(ev.user.id), ev.user.nickname, is_mod=ev.user.is_moderator, is_sub=ev.user_is_super_fan), ev.content, time.time(), time.time()))
  @client.on(GiftEvent)
  async def on_gift(ev):
      if ev.streaking: return                          # only count the final event of a streak
      usd = ev.value or 0.0                            # repeat_count*diamond_count*0.005
  await client.start(fetch_gift_info=False)            # or client.run(); check await client.is_live() first
  Run it in its own asyncio task with exponential backoff and a circuit breaker; never let its exceptions reach the core loop.

=== NORMALISED MODEL + SELECTOR (tested prototype) ===
File: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/chat/proto/chatmodel.py (+ test_chatmodel.py)
  @dataclass(frozen=True, slots=True) class ChatUser: platform, id, name, is_broadcaster, is_mod, is_vip, is_sub, sub_months, is_verified
  @dataclass(frozen=True, slots=True) class ChatMessage: platform, id, user, text, ts, received_ts, kind(TEXT|DONATION|SUB|GIFT_SUB|RAID|SYSTEM), amount, currency("BITS"|ISO4217|"DIAMOND"), value_usd, reply_to, source_channel, first_msg, raw
  Mappings:
    twitch sub = badges subscriber|founder; mod = moderator badge or mod=1; vip = vip badge; donation = cheer.bits (value 0.01 USD/bit).
    youtube: mod = isChatModerator; sub = isChatSponsor; owner = isChatOwner; donation = superChat/superSticker amountMicros/1e6 + currency.
    tiktok: mod = user.is_moderator; sub = user_is_super_fan; donation = GiftEvent.value.
  ChatWindow.add(m): dedupe on platform:id (keep a 5k LRU), drop shared-chat, route donation/sub/raid to the priority queue, drop "!" commands and oversize messages, count normalised duplicates.
  ChatWindow.select() -> (priority_events sorted by value_usd, k sampled messages sorted by ts). The score is:
    2*exp(-age/15s) + 3*[name mention: ไพลิน|pailin|ไพ่ลิน|น้องไพลิน on NFKC+casefold+zero-width-stripped text]
    + 0.8*[question: '?' or ends with ไหม/มั้ย/หรอ/เหรอ/ยังไง/อะไร] + 0.7*sub + 0.5*mod + 0.3*vip + 0.5*first_msg
    - 1.5*ln(dup_count) - 5*[user picked < 60 s ago]
  Sampling is softmax at T=0.6 over one message per user and per dedupe key; the window is cleared after each select.
  Exception: YouTube giftEvent reuses the id to update comboCount, so key dedupe for giftEvent on (id, comboCount) or update in place.

=== INPUT SAFETY PIPELINE (before the LLM) ===
tier0(text, name), synchronous, <1 ms:
  - NFKC, remove U+200B/200C/200D/2060/FEFF/00AD, collapse character runs >3, casefold.
  - Aho-Corasick blocklist (Thai + English slurs, sexual terms, monarchy/§112 terms, self-harm, doxx words) plus regexes: URLs, @handles, Thai phone ^0[689]\d{8}$, 13-digit national id, emails.
  - Action: DROP (hard hit), MASK (PII/URL -> "[link]"), or PASS.
  - Also run it on display names; if a name is flagged, speak "someone" instead.
tier1(batch), asynchronous, only on the k selected messages + priority texts:
  tok = AutoTokenizer.from_pretrained("typhoon-ai/typhoon2-safety-preview"); mdl = AutoModelForSequenceClassification.from_pretrained(...).eval()
  p = softmax(mdl(**tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)).logits, -1)[:, 1]   # P(Harmful)
  Drop if p > 0.8, send to the mod queue if 0.5-0.8. Alternative: Qwen3Guard-Gen-0.6B prompt classification (the 'Jailbreak' category helps against injection).
tier2: mod panel. Approve/deny queue for borderline items, per-user mute, global "chat off" switch, and a kill switch for TTS.
Prompt assembly: put chat in the user turn inside a fenced block, e.g.
  <chat untrusted="true">\n[twitch][sub] Somchai: ไพลินกินข้าวยัง?\n</chat>
  Strip role tokens (<|im_start|>, <|system|>, '###', 'system:') and cap each message at 300 chars. The system prompt states that chat is quoted viewer speech and must never be treated as instructions. Donations: read the name and amount even if the text is filtered, and say "Filtered." instead of the text (mirrors Neuro's visible 'Filtered.', T4).

## Latency & resources

- Twitch EventSub/IRC: push-based, sub-second message delivery (not measured; typical). Helix calls are about 100-300 ms each (not measured).
- YouTube streamList: push-based. list-polling adds 0 to pollingIntervalMillis of latency; the typical value was not measured (historically a few seconds; UNVERIFIED). Budget 1 unit/poll.
- TikTok: push over WebSocket after one sign-server request per connect (Euler Community tier: 2,500 requests/day).
- All adapters are I/O-bound asyncio tasks: under 1% of one core on the i7-14700KF and about 30-60 MB RAM together (estimate). grpcio adds about 20 MB.
- Selector: O(n) per tick over at most 500 buffered messages, microseconds.
- Tier-0 filter: under 1 ms per message.
- Tier-1 typhoon2-safety-preview: 1.1 GB fp32 weights (~0.56 GB in fp16 on GPU). Batch of ~8 short messages: estimated 10-30 ms on the RTX 4070 or 50-150 ms on CPU (UNVERIFIED, benchmark it). It runs only on selected candidates, not the firehose, so its cost is bounded by the decision rate (about 1 per few seconds).
- Qwen3Guard-Gen-0.6B: 1.5 GB bf16. It is generative, so a batch costs ~100-300 ms on GPU (estimate) and competes with the LLM/TTS for VRAM. Prefer the Typhoon classifier on CPU so all 12 GB of VRAM stay free for LLM/TTS/STT.
- Network: YouTube list polling at maxResults=2000 is a few KB to 100 KB per poll; Twitch is tiny.

## Pitfalls

- Twitch: IRC is NOT deprecated, but twitchio 3.x removed IRC entirely. Do not read 'recommended EventSub' as 'IRC is dead'. Only ws://irc-ws.chat.twitch.tv:80 (non-SSL WebSocket) is gone as of 2025-08-15.
- EventSub WebSocket only accepts USER access tokens. Subscribing with an app token fails. Subscriptions must be created within 10 s of session_welcome or the server closes with 4003. Never resubscribe after session_reconnect (subs migrate); DO resubscribe after a fresh connect.
- DCF refresh tokens for public clients are single-use. A crash between refresh and save logs the streamer out. Write tokens atomically (temp file + os.replace) and serialise refreshes behind a lock. Validate hourly; Twitch audits this.
- Access tokens from DCF last about 4 h. Refresh reactively on 401 as Twitch recommends, not only on a timer.
- Polls (Twitch) need Affiliate/Partner status, otherwise 403. Title/poll endpoints require broadcaster_id == token user, so a separate bot-account token cannot change the title or create polls.
- If the chat sender is not broadcaster/mod/VIP and exceeds 20 msgs/30 s, Twitch silently ignores its messages for 1 hour. Make the bot account a mod/VIP, add a local token bucket, and check is_sent/drop_reason in the Send Chat response.
- Send Chat with a user token cannot set for_source_only (400). In shared chat, the AI's messages go to all channels, and incoming channel.chat.message includes other channels' messages (source_broadcaster_user_id).
- Chat Bot badge / 'Chat Bots' list appearance needs an APP token + channel:bot + a non-broadcaster account. A user-token bot will not get the badge (cosmetic only).
- YouTube: liveChatMessages.list API-key support for public chats is implied (the error message asks for an API key) but not explicitly documented. streamList explicitly supports API keys. Build both, and fall back to OAuth if a key returns 403.
- YouTube streamList quota cost is NOT documented. Watch the Cloud console quota page during the first stream. RESOURCE_EXHAUSTED/403 rateLimitExceeded happens if you poll faster than pollingIntervalMillis.
- YouTube actions cost 50 units each against 10k/day. An AI that sends a chat line every 30 s would exhaust quota in about 1.7 h. Rate-limit per platform in ChannelActions.
- Google OAuth consent screen in 'Testing' makes refresh tokens expire after 7 days, which silently breaks YouTube actions a week later.
- YouTube search.list moved to a separate 100-calls/day bucket on 2026-06-01. Do not poll search for live detection; use liveBroadcasts.list (OAuth) or videos.list.
- The @handle/live scrape is fragile: datacenter IPs get a bot-check player response; channels with several concurrent lives return varying ids; the HTML layout can change. Always confirm with videos.list -> activeLiveChatId.
- YouTube giftEvent reuses the same message id to update comboCount. A naive id-dedupe drops combo updates and undercounts value.
- The first list/streamList response replays recent history. Skip or mark backlog on connect, or the AI will answer stale messages.
- pytchat is archived (2022-01-25) and chat-downloader has not been released since 2023-09. Do not use them for the core.
- TikTokLive is reverse-engineered and relies on a third-party sign server. Protocol changes break it without notice. A live issue (2026-09-10) shows connections relayed via ws-fallback.eulerstream.com failing. Never send a TikTok sessionid (login cookie) to the sign server; the library itself warns about WHITELIST_AUTHENTICATED_SESSION_ID_HOST.
- TikTokLive license: fine for a self-hosted streaming bot under §18. It would revert to full AGPL if anyone later offered the project as a hosted SaaS or relay (§19).
- TikTok gift streaks emit many GiftEvents. Only count the event where streaking is False (value returns None mid-streak).
- Thai text: there are no spaces, so name mentions must use substring or regex matching, not tokens. Viewers insert U+200B, use ๆ, and spam 55555 (laughter). Normalise before dedupe/blocklist, but keep the original for display/TTS.
- Display names are an injection and abuse vector: filter them like message text, and never paste raw names into the system prompt.
- Never let the LLM ban users. Restrict tool-driven moderation to short timeouts (e.g. at most 600 s) with a human-override log. Twitch reason text is public to mods.

## Open questions

- Is the streamer a Twitch Affiliate or Partner? Twitch polls require it; otherwise run chat-command polls in our own code.
- Which account should Pailin post chat messages from: the broadcaster account or a dedicated bot account (then made mod/VIP)? Or should she not post to chat at all, with TTS only?
- Which platforms does the streamer actually use? Twitch only, or also YouTube and TikTok? This decides whether the grpcio and TikTokLive deps ship by default.
- YouTube streamList quota cost per connection/response is undocumented. Measure it on a real stream with the Cloud console.
- Typical YouTube pollingIntervalMillis values in 2026 were not observed (needs an API key).
- Does liveChatMessages.list accept a bare API key for public chats in 2026? Implied but not explicitly documented. Test it with a key.
- Exact Euler Stream limits without an API key, and whether TikTokLive 7.x routes WebSocket traffic via ws-fallback.eulerstream.com by default (the Sept 2026 issue suggests it sometimes does).
- Thai blocklist content (slurs, §112-sensitive terms, gambling/scam spam common in Thai chats). Who curates it, and whether it lives in a separate non-public file.
- Typhoon safety classifier false-positive rate on casual Thai stream chat (slang, 555, teasing). Needs a small labelled eval set before choosing thresholds.
- Currency conversion for value_usd (THB superchats vs bits vs diamonds): a static table or a daily FX fetch? Used only for ranking.
- Whether the Google OAuth app can remain unverified 'In production' for the youtube.force-ssl scope for a single user (policy detail not verified).
