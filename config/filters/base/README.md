# Tier-0 base filter lists

Committed word and phrase lists for the tier-0 `TextFilter` (`aivtube.safety.keyword`,
ARCHITECTURE.md §7). `safety.base_lists` in `config/defaults.toml` points here. Every
`*.toml` file in this folder is loaded, in file-name order.

These lists are deliberately **conservative**: they apply to every stream and every
character. Ambiguous everyday words stay out (for example `ควาย` "buffalo", `กะหรี่` "curry",
`ม็อบ` "mob" in games, the Latin prefix `porn` in Thai names such as Pornchai). Add local
words elsewhere:

- `config/filters/private/*.toml` (gitignored): the streamer's own additions.
- `safety.platform_overlays.<platform>`: one TOML file per platform.
- `characters/<id>/filters.toml`: one character's overlay.

Layers stack as base < private < platform < character: the platform layer applies to that
platform's chat only, the character layer to that character only. An `allow` phrase listed
in any active layer is never a hit, whichever layer denies it, except in fail-closed
categories (`monarchy_112`), which no allow can loosen.

## File format

A file may use the compact form, the table form, or both.

```toml
category  = "slur"        # file-level category for the compact arrays
token     = ["..."]       # words/phrases matched on newmm word boundaries (Thai-safe)
substring = ["..."]       # unambiguous strings matched anywhere
regex     = ['...']       # Python regexes over the normalised (lower-case) text
allow     = ["..."]       # phrases that are never a hit

[[allow]]
text = "หีบ"

[[deny]]
category = "slur"         # optional when the file sets category
match = "token"           # token (default) | substring | regex
text = "..."

[[replace]]               # a hard-coded speech patch
match = "regex"           # regex (default) | substring (case-insensitive literal)
text = "(?i)as an ai language model,?\\s*"
with = ""
directions = ["out"]      # default ["out"]; any of in, out, tool, memory, name, game
```

Matching runs on normalised text: NFKC, zero-width characters removed, `pythainlp`
normalisation, casefold, character runs longer than 2 shortened to 2 (`ควยยยย` → `ควยย`),
Thai digits mapped to Arabic. Write entries in their plain dictionary spelling.

- `token` entries only hit when the match starts and ends on a newmm word boundary
  (fail-closed categories excepted: their Thai entries match as plain substrings), so
  `หี` does not fire inside `หีบ` and `ฆ่า` alone is not listed (game talk such as
  `ฆ่ามอนสเตอร์` must pass). Multi-word phrases work (`ไปตายซะ`), as do stretched endings
  (`สัสส`) and letter-spaced spellings (`ค ว ย`, `ค.ว.ย`).
- `substring` entries also match letter-spaced (`f u c k`) and leet (`n1gg3r`) spellings.
  Use them only for strings that are never part of an innocent word.
- `regex` entries see the normalised text (lower case, Thai digits as `0`-`9`).
- `pii` and `injection` rules run on the display text instead, so the filter can mask
  (`pii`, shown as `[ลิงก์]`) or strip (`injection`) exactly the matched span. They accept
  only `substring` and `regex`.

## Categories and verdicts

| Category | Chat input / names | Output, tool, memory, game |
|---|---|---|
| slur, sexual, doxx, violent_extreme | DROP | BLOCK |
| self_harm | DROP and a panel alert | BLOCK |
| monarchy_112 | DROP (fail-closed) | BLOCK (fail-closed) |
| gambling_scam | DROP | BLOCK |
| politics | REVIEW (`safety.politics`; strict mode blocks) | REVIEW (strict mode: BLOCK) |
| pii | MASK | BLOCK |
| injection | stripped | not applied |

Reload after editing with the panel's **Reload filters** action (`FILTER_RELOAD`). A file
with an error is rejected and the previous lists stay active.
