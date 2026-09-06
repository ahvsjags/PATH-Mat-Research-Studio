$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    python -m venv (Join-Path $Root ".venv")
    & $Python -m pip install -r (Join-Path $Root "requirements.txt")
    & $Python -m playwright install chromium
}
$env:PYTHONPATH = "$Root;$Root\vendor"
$env:PYTHONIOENCODING = "utf-8"
& $Python (Join-Path $Root "app.py") --host 0.0.0.0 --port 7860 --share
