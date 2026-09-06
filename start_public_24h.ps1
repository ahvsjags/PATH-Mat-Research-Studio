param(
    [int]$Hours = 24
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Cloudflared = "C:\Users\admin\AppData\Local\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
$Runtime = Join-Path $Root "runtime_24h"
$Assets = Join-Path $Root "assets"
$Output = Resolve-Path (Join-Path $Root "..\..\output")

New-Item -ItemType Directory -Force -Path $Runtime, $Assets | Out-Null
$AppOut = Join-Path $Runtime "app.stdout.log"
$AppErr = Join-Path $Runtime "app.stderr.log"
$TunnelOut = Join-Path $Runtime "cloudflared.stdout.log"
$TunnelErr = Join-Path $Runtime "cloudflared.stderr.log"
foreach ($file in @($AppOut, $AppErr, $TunnelOut, $TunnelErr)) {
    if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force }
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Python runtime not found: $Python" }
if (-not (Test-Path -LiteralPath $Cloudflared)) { throw "cloudflared not found: $Cloudflared" }

$StartedAt = Get-Date
$ExpiresAt = $StartedAt.AddHours($Hours)
$AppProcess = $null
$TunnelProcess = $null

function Start-PathMatApp {
    return Start-Process -FilePath $Python `
        -ArgumentList @("app.py", "--host", "127.0.0.1", "--port", "7860", "--no-block") `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $AppOut -RedirectStandardError $AppErr
}

try {
    $AppProcess = Start-PathMatApp
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 1
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:7860/" -UseBasicParsing -TimeoutSec 4
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        if ($AppProcess.HasExited) { throw "PATH-Mat app exited during startup. See $AppErr" }
    }
    if (-not $ready) { throw "PATH-Mat app did not become ready within 90 seconds." }

    $TunnelProcess = Start-Process -FilePath $Cloudflared `
        -ArgumentList @("tunnel", "--url", "http://127.0.0.1:7860", "--no-autoupdate") `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $TunnelOut -RedirectStandardError $TunnelErr

    $PublicUrl = $null
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 1
        $combined = ""
        if (Test-Path -LiteralPath $TunnelOut) { $combined += Get-Content -LiteralPath $TunnelOut -Raw }
        if (Test-Path -LiteralPath $TunnelErr) { $combined += Get-Content -LiteralPath $TunnelErr -Raw }
        $match = [regex]::Match($combined, "https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+){2,}\.trycloudflare\.com")
        if ($match.Success) { $PublicUrl = $match.Value; break }
        if ($TunnelProcess.HasExited) { throw "Cloudflare tunnel exited during startup. See $TunnelErr" }
    }
    if (-not $PublicUrl) { throw "Cloudflare public URL was not created within 90 seconds." }

    $QrPath = Join-Path $Assets "pathmat_console_qr.png"
    $env:PATHMAT_PUBLIC_URL = $PublicUrl
    $env:PATHMAT_QR_PATH = $QrPath
    & $Python -c "import os,qrcode; qrcode.make(os.environ['PATHMAT_PUBLIC_URL']).save(os.environ['PATHMAT_QR_PATH'])"
    Set-Content -LiteralPath (Join-Path $Assets "share_url.txt") -Value $PublicUrl -Encoding utf8
    Copy-Item -LiteralPath $QrPath -Destination (Join-Path $Output "PATH-Mat_Research_Console_扫码访问.png") -Force

    $State = [ordered]@{
        public_url = $PublicUrl
        started_at = $StartedAt.ToString("o")
        expires_at = $ExpiresAt.ToString("o")
        duration_hours = $Hours
        app_pid = $AppProcess.Id
        tunnel_pid = $TunnelProcess.Id
        status = "active"
    }
    $State | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Assets "public_session.json") -Encoding utf8

    Write-Output "PUBLIC_URL=$PublicUrl"
    Write-Output "STARTED_AT=$($StartedAt.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
    Write-Output "EXPIRES_AT=$($ExpiresAt.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
    Write-Output "QR_PATH=$QrPath"

    $nextHealthCheck = Get-Date
    while ((Get-Date) -lt $ExpiresAt) {
        Start-Sleep -Seconds 15
        if ($AppProcess.HasExited) {
            $AppProcess = Start-PathMatApp
            Start-Sleep -Seconds 8
        }
        if ($TunnelProcess.HasExited) { throw "Cloudflare tunnel stopped before the 24-hour expiry." }
        if ((Get-Date) -ge $nextHealthCheck) {
            try { Invoke-WebRequest -Uri $PublicUrl -UseBasicParsing -TimeoutSec 15 | Out-Null } catch {}
            $nextHealthCheck = (Get-Date).AddMinutes(3)
        }
    }
}
finally {
    if ($TunnelProcess -and -not $TunnelProcess.HasExited) { Stop-Process -Id $TunnelProcess.Id -Force }
    if ($AppProcess -and -not $AppProcess.HasExited) { Stop-Process -Id $AppProcess.Id -Force }
    $StatePath = Join-Path $Assets "public_session.json"
    if (Test-Path -LiteralPath $StatePath) {
        $finalState = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        $finalState.status = "expired"
        $finalState | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding utf8
    }
}
