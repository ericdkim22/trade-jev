@echo off
rem Daily live session (Windows scheduled task "trade-jev-live"): IBKR feed, MNQ front month, Jev on.
cd /d %~dp0..
if not exist logs mkdir logs
set PYTHONUTF8=1
echo ==== %date% %time% >> logs\live.log
.venv\Scripts\python -m trade_jev.live --commission 0.62 --no-open >> logs\live.log 2>&1
