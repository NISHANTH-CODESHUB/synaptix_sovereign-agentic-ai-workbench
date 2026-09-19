# Prove the sovereignty claim. RUN AS ADMINISTRATOR.
#
#   .\scripts\airgap.ps1 -On    block all outbound traffic from python + ollama
#   .\scripts\airgap.ps1 -Off   remove the rules again
#   .\scripts\airgap.ps1        show current state
#
# The simplest proof is still to unplug the cable / turn off Wi-Fi — this script
# is for when you want the machine online for everything EXCEPT the workbench,
# which is the more honest demonstration: the box has a route, and we still
# cannot use it.
param([switch]$On, [switch]$Off)

$root = Split-Path $PSScriptRoot -Parent
$targets = @(
  (Join-Path $root ".venv\Scripts\python.exe"),
  (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
  (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe")
) | Where-Object { Test-Path $_ }

$prefix = "KWS-AirGap"

function Show {
  $rules = Get-NetFirewallRule -DisplayName "$prefix*" -ErrorAction SilentlyContinue
  if ($rules) {
    Write-Host "`n  Air-gap rules ACTIVE:" -ForegroundColor Green
    $rules | ForEach-Object { Write-Host "    $($_.DisplayName)  [$($_.Action)]" }
  } else {
    Write-Host "`n  No air-gap rules present — this machine can reach the internet." -ForegroundColor Yellow
  }
  Write-Host ""
}

if ($On) {
  foreach ($t in $targets) {
    $name = "$prefix - $(Split-Path $t -Leaf)"
    Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue |
      Remove-NetFirewallRule -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName $name -Direction Outbound -Program $t `
      -Action Block -Profile Any -RemoteAddress Internet | Out-Null
    Write-Host "  blocked outbound: $t" -ForegroundColor Green
  }
  Write-Host "`n  Loopback and LAN still work. The canary in the UI should turn" -ForegroundColor DarkGray
  Write-Host "  green within 5 seconds." -ForegroundColor DarkGray
  Show
}
elseif ($Off) {
  Get-NetFirewallRule -DisplayName "$prefix*" -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule
  Write-Host "  air-gap rules removed." -ForegroundColor Yellow
  Show
}
else { Show }
