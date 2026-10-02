@echo off
rem Starts SolTrade through run.ps1, whatever the PowerShell execution policy. Arguments
rem (e.g. --dry-run) are passed on. Pauses when something needs reading before the window closes.
rem Windows PowerShell by its full path, since its folder isn't always on PATH.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
set "code=%errorlevel%"
if not "%code%"=="0" pause
exit /b %code%
