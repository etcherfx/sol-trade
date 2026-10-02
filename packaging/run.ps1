# Starts SolTrade from this folder: installs uv if it's missing (after asking), creates the
# config files on the first run, then runs the bot. Arguments (e.g. --dry-run) are passed on.
# Exits 2 after creating the config files, so run.cmd keeps the window open to show why.
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
    Write-Host 'SolTrade runs with uv (https://docs.astral.sh/uv/), which is not installed.'
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
foreach ($file in 'config.json', '.env') {
    if (-not (Test-Path -LiteralPath $file)) {
        Copy-Item -LiteralPath "$file.sample" -Destination $file
        $created += $file
    }
}
if ($created.Count -gt 0) {
    Write-Host "Created $($created -join ' and ') from the samples. Set SOLTRADE_PRIVATE_KEY in .env and the tokens to trade"
    Write-Host "(secondary_mints) in config.json, then run this again. See the README's Getting started."
    exit 2
}

& uv run --frozen main.py @args
exit $LASTEXITCODE
