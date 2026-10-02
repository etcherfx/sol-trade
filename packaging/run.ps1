# Starts the project from this folder: installs uv if it's missing (after asking), creates the
# config files on the first run, then runs it with the locked dependencies. Arguments are
# passed on. Exits 2 after creating the config files, so run.cmd keeps the window open to
# show why.

# --- project settings ---------------------------------------------------------
$Project = 'SolTrade'
# Sample file = file it's copied to on the first run.
$Configs = [ordered]@{ 'config.json.sample' = 'config.json'; '.env.sample' = '.env' }
# Shown after the first run creates the config files.
$FirstRunHint = "Set SOLTRADE_PRIVATE_KEY in .env and the tokens to trade (secondary_mints) in config.json, then run this again. See the README's Getting started."
$Entry = 'main.py'
# -----------------------------------------------------------------------------

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Find-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) { return $true }
    # where the official installer puts uv, for a window that opened before it was installed
    $bin = Join-Path $env:USERPROFILE '.local\bin'
    if (Test-Path -LiteralPath (Join-Path $bin 'uv.exe')) {
        $env:Path = "$bin;$env:Path"
        return $true
    }
    return $false
}

if (-not (Find-Uv)) {
    Write-Host "$Project runs with uv (https://docs.astral.sh/uv/), which is not installed."
    $answer = Read-Host 'Install it now with the official installer? [y/N]'
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host 'uv is required. Install it from https://docs.astral.sh/uv/ and run this again.'
        exit 1
    }
    # in a separate process, as Astral documents it, with the PowerShell running this script
    & (Get-Process -Id $PID).Path -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
    if (-not (Find-Uv)) {
        Write-Host "uv was installed but can't be found. Open a new window and run this again."
        exit 1
    }
}

$created = @()
foreach ($sample in $Configs.Keys) {
    $target = $Configs[$sample]
    if (-not (Test-Path -LiteralPath $target)) {
        Copy-Item -LiteralPath $sample -Destination $target
        $created += $target
    }
}
if ($created.Count -gt 0) {
    Write-Host "Created $($created -join ' and ') from the samples. $FirstRunHint"
    exit 2
}

& uv run --frozen $Entry @args
exit $LASTEXITCODE
