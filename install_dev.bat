@echo off
setlocal
cd /d "%~dp0"
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b %errorlevel%
python -m shpb_processor.sample_data --output examples
python -m pytest
