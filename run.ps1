# SynaptiX — start the workbench.
#   .\run.ps1              start server on http://127.0.0.1:8080
#   .\run.ps1 -Rebuild     wipe the DB and rebuild the demo corpus first
param([switch]$Rebuild)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
$env:HF_HUB_OFFLINE = "1"          # embedder is cached in data/models/embed
$env:TRANSFORMERS_OFFLINE = "1"
$env:DO_NOT_TRACK = "1"
$env:ANONYMIZED_TELEMETRY = "False"
$env:OLLAMA_KEEP_ALIVE = "15m"
# VRAM economics (design dossier section 5): quantising the KV cache and enabling
# flash attention took qwen3:4b from 4.1 GB to 3.4 GB resident on a 4094 MiB card,
# which moved it from 55% to 67% GPU offload and roughly doubled decode speed.
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"

Write-Host "`n  SYNAPTIX — sovereign AI workbench" -ForegroundColor White
Write-Host "  ------------------------------------" -ForegroundColor DarkGray

# --- ollama up? ---
try {
  Invoke-WebRequest "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null
  Write-Host "  ollama    : running" -ForegroundColor Green
} catch {
  Write-Host "  ollama    : starting bare server (NOT the tray app)" -ForegroundColor Yellow
  # The desktop tray app ("ollama app.exe") runs an hourly update checker that
  # calls out to the internet. The bare server does not. Never launch the tray
  # app on a machine you are calling air-gapped.
  Get-Process "ollama app" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
  Start-Sleep -Seconds 8
}

if ($Rebuild) {
  Write-Host "  rebuild   : wiping DB and regenerating demo corpus" -ForegroundColor Yellow
  Remove-Item data\kws.db* -Force -ErrorAction SilentlyContinue
  & .\.venv\Scripts\python.exe scripts\make_demo_corpus.py
}

$tags = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
Write-Host "  models    : " -NoNewline; ($tags.models | ForEach-Object { $_.name }) -join ", " | Write-Host -ForegroundColor Cyan
Write-Host "  workbench : http://127.0.0.1:8080" -ForegroundColor Cyan
Write-Host "  air-gap   : run scripts\airgap.ps1 as Administrator to prove zero egress`n" -ForegroundColor DarkGray

& .\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8080
