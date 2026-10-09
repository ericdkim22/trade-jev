@echo off
rem Daily live session (Windows scheduled task "trade-jev-live"): IBKR feed, MNQ front month, Jev on.
cd /d %~dp0..
if not exist logs mkdir logs
set PYTHONUTF8=1
rem Stopping the task (or its 14 h limit) kills cmd but not python: end yesterday's session so port 8765 is free.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'trade_jev\.live' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo ==== %date% %time% >> logs\live.log
.venv\Scripts\python -m trade_jev.live --commission 0.62 --no-open --host 0.0.0.0 --variants features >> logs\live.log 2>&1
