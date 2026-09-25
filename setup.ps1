# setup.ps1 — only bootstraps uv; all logic lives in `aivtube setup` (Python, testable, resumable)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { irm https://astral.sh/uv/install.ps1 | iex; $env:Path = "$env:USERPROFILE\.local\bin;$env:Path" }
uv python install 3.12
uv sync --frozen --extra pc
uv run aivtube setup @args
