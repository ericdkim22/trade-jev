@echo off
rem Nightly variant research (Windows scheduled task "trade-jev-research"): see src/trade_jev/research.py.
cd /d %~dp0..
if not exist logs mkdir logs
set PYTHONUTF8=1
echo ==== %date% %time% >> logs\research.log
.venv\Scripts\python -m trade_jev.research nightly >> logs\research.log 2>&1
echo ---- news fade >> logs\research.log
.venv\Scripts\python scripts\news_nq.py nightly >> logs\research.log 2>&1
