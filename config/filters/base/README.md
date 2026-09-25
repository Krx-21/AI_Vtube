# Tier-0 base filter lists

Committed word and phrase lists for the tier-0 `TextFilter` (ARCHITECTURE.md §7).
`safety.base_lists` in `config/defaults.toml` points here.

- The safety module (WP8) owns this folder and defines the list file format.
- Put operator-private additions in `config/filters/private/`. That folder is gitignored.
- Per-character overlays live in `characters/<id>/filters.toml`.
