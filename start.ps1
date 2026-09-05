# One-shot setup and launch (Windows).
#
#   .\start.ps1                 set everything up and open the app
#   .\start.ps1 -NoBrowser      do not open a browser at the end
#   .\start.ps1 -NoAutoUpdate   do not arm the background update watcher
#
# Models run through Docker Model Runner - a Docker Desktop feature that
# executes them natively on this machine's GPU. On Windows with an AMD card
# that means the Vulkan llama.cpp backend, shipped with current Docker
# Desktop. Safe to re-run: it never overwrites .env and never re-downloads
# a model it already has.
#
# Written for Windows PowerShell 5.1, the one preinstalled on every Windows
# box. No PowerShell 7 syntax in here (no &&, no ternary, no ??), and ASCII
# only - 5.1 reads unBOMed files as the ANSI codepage, so one curly quote
# would turn into mojibake or a parse error.

[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$NoAutoUpdate
)

$ErrorActionPreference = 'Continue'

$Root = $PSScriptRoot
Set-Location $Root

if ($Env:OS -ne 'Windows_NT') {
    Write-Host 'This script is the Windows entry point. On macOS or Linux run ./start.sh instead.'
    exit 1
}

# --- output helpers ---------------------------------------------------------
function Step([string]$Msg) {
    Write-Host ''
    Write-Host "==> $Msg" -ForegroundColor Cyan
}
function Ok([string]$Msg) {
    Write-Host -NoNewline '    '
    Write-Host -NoNewline 'ok   ' -ForegroundColor Green
    Write-Host $Msg
}
function Warn([string]$Msg) {
    Write-Host -NoNewline '    '
    Write-Host -NoNewline 'warn ' -ForegroundColor Yellow
    Write-Host $Msg
}
function Fail([string]$Msg) {
    Write-Host ''
    Write-Host "ERROR $Msg" -ForegroundColor Red
    Write-Host ''
    exit 1
}
function Have([string]$Name) {
    return ($null -ne (Get-Command $Name -ErrorAction SilentlyContinue))
}

# --- Docker -------------------------------------------------------------------
# A fresh Docker Desktop install puts the CLI in a directory this session's
# PATH has not seen, so widen PATH before every probe.
$DesktopExeCandidates = @(
    (Join-Path $Env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
    (Join-Path $Env:LOCALAPPDATA 'Programs\Docker\Docker\Docker Desktop.exe')
)
# A 32-bit PowerShell host on 64-bit Windows sees Program Files (x86) as
# ProgramFiles; the real install lives under ProgramW6432.
if ($Env:ProgramW6432) {
    $DesktopExeCandidates += (Join-Path $Env:ProgramW6432 'Docker\Docker\Docker Desktop.exe')
}

function Add-DockerPath {
    $candidates = @(
        (Join-Path $Env:ProgramFiles 'Docker\Docker\resources\bin'),
        (Join-Path $Env:LOCALAPPDATA 'Programs\Docker\Docker\resources\bin')
    )
    if ($Env:ProgramW6432) {
        $candidates += (Join-Path $Env:ProgramW6432 'Docker\Docker\resources\bin')
    }
    foreach ($d in $candidates) {
        if ((Test-Path $d) -and (-not ($Env:Path -like "*$d*"))) {
            $Env:Path = $Env:Path + ';' + $d
        }
    }
}

function Test-DockerReady {
    try { docker info *> $null } catch { return $false }
    return ($LASTEXITCODE -eq 0)
}

function Install-DockerDesktop {
    # Already installed but this shell just cannot see it? Nothing to install.
    foreach ($c in $DesktopExeCandidates) {
        if (Test-Path $c) { return }
    }
    Step 'Installing Docker Desktop'
    if (-not (Have 'winget')) {
        Fail ("Docker is not installed and winget is not available on this machine.`n" +
              '    Download Docker Desktop from https://www.docker.com/products/docker-desktop' + "`n" +
              '    install it, open it once to accept the licence, then run start.ps1 again.')
    }
    Warn 'winget will raise a UAC (administrator) prompt to finish this install.'
    winget install -e --id Docker.DockerDesktop --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Fail ("Docker Desktop install failed. Install it manually from`n" +
              '    https://www.docker.com/products/docker-desktop then run start.ps1 again.')
    }
    Warn 'If Docker is still not found after this, log out and back in (or restart):'
    Warn 'a fresh install needs that for PATH and the docker-users group to take effect.'
}

function Start-DockerDesktop {
    Step 'Starting Docker Desktop'
    $exe = $null
    foreach ($c in $DesktopExeCandidates) {
        if (($null -eq $exe) -and (Test-Path $c)) { $exe = $c }
    }
    if ($null -eq $exe) {
        Fail ("The Docker daemon is not running and Docker Desktop was not found.`n" +
              '    Start Docker Desktop from the Start menu, then run start.ps1 again.')
    }
    try { Start-Process $exe | Out-Null } catch {
        Fail 'Could not launch Docker Desktop. Open it from the Start menu, then run start.ps1 again.'
    }
    Write-Host -NoNewline '    waiting for the Docker daemon (first launch asks you to accept a licence, and may ask to enable WSL 2)'
    $waited = 0
    while (-not (Test-DockerReady)) {
        if ($waited -ge 180) {
            Write-Host ''
            Fail ("Docker did not become ready in 3 minutes.`n" +
                  '    Open Docker Desktop, finish any licence or WSL 2 prompt it shows, then run start.ps1 again.')
        }
        Write-Host -NoNewline '.'
        Start-Sleep -Seconds 3
        $waited += 3
    }
    Write-Host ''
}

Step 'Checking Docker'
Add-DockerPath
if (-not (Have 'docker')) {
    Install-DockerDesktop
    Add-DockerPath
    if (-not (Have 'docker')) {
        Fail ("Docker is installed but the 'docker' command is not on this shell's PATH yet.`n" +
              '    Close this window, open a new one (or log out and back in), and run start.ps1 again.')
    }
}
if (-not (Test-DockerReady)) {
    Start-DockerDesktop
}
if (-not (Test-DockerReady)) {
    Fail 'Docker is still not reachable.'
}
$dockerVersion = ''
try { $dockerVersion = ('' + (docker version --format '{{.Server.Version}}' 2>$null)).Trim() } catch { }
if ($dockerVersion -eq '') { $dockerVersion = 'running' }
Ok "Docker $dockerVersion"

# Compose v2 ships inside Docker Desktop; v1 is a legacy standalone binary.
$script:ComposeV2 = $true
$ComposeCmd = 'docker compose'
docker compose version *> $null
if ($LASTEXITCODE -ne 0) {
    # No v1 fallback on purpose: the health checks and the running-stack
    # probe use v2-only flags, so v1 would fail later and less clearly.
    Fail 'Docker Compose v2 is required. Update Docker Desktop.'
}
function Compose {
    if ($script:ComposeV2) { docker compose @args } else { docker-compose @args }
}

# --- .env ---------------------------------------------------------------------
# All writes use -Encoding ASCII: a UTF-8 BOM at the top of .env would glue
# itself to the first key name and docker compose would never see that value.
Step 'Configuration'
if (-not (Test-Path '.env')) {
    if (-not (Test-Path '.env.example')) {
        Fail '.env.example is missing - is this a complete checkout?'
    }
    Copy-Item '.env.example' '.env' -ErrorAction Stop
    Ok 'created .env from .env.example'
} else {
    Ok '.env already exists (left untouched)'
}

# .env I/O is lossless UTF-8 without a BOM. -Encoding ASCII would silently
# turn any byte above 127 into '?' on every rewrite, and a BOM would glue
# itself to the first key so compose never sees that value. Writes are
# checked: a read-only file or Controlled Folder Access must fail loudly.
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
function Read-EnvLines {
    return [System.IO.File]::ReadAllLines((Join-Path $Root '.env'))
}
function Write-EnvLines([string[]]$Lines) {
    try {
        [System.IO.File]::WriteAllLines((Join-Path $Root '.env'), $Lines, $Utf8NoBom)
    } catch {
        Fail ("could not write .env - check it is not read-only and that " +
              "Controlled Folder Access is not blocking this folder. ($_)")
    }
}
function Get-EnvValue([string]$Key) {
    foreach ($line in @(Read-EnvLines)) {
        if ($line -match ('^' + [regex]::Escape($Key) + '=(.*)$')) { return $Matches[1] }
    }
    return ''
}
function Set-EnvValue([string]$Key, [string]$Value) {
    $lines = @(Read-EnvLines)
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match ('^' + [regex]::Escape($Key) + '=')) {
            $lines[$i] = "$Key=$Value"
            $found = $true
        }
    }
    if (-not $found) { $lines += "$Key=$Value" }
    Write-EnvLines $lines
}
function Set-EnvIfBlank([string]$Key, [string]$Value) {
    if ((Get-EnvValue $Key) -eq '') { Set-EnvValue $Key $Value }
}

if ((Get-EnvValue 'NEO4J_PASSWORD') -eq '') {
    # Alphanumeric only: a '/' in the password breaks NEO4J_AUTH=neo4j/<pw>,
    # and '$' breaks compose interpolation. 48 hex characters satisfy both,
    # from the OS crypto RNG (Get-Random is not one).
    $rng = New-Object System.Security.Cryptography.RNGCryptoServiceProvider
    $buf = New-Object byte[] 24
    $rng.GetBytes($buf)
    $pw = ($buf | ForEach-Object { $_.ToString('x2') }) -join ''
    if ($pw.Length -ne 48) { Fail 'Could not generate a database password.' }
    Set-EnvValue 'NEO4J_PASSWORD' $pw
    Ok 'generated a random NEO4J_PASSWORD'
} else {
    Ok 'NEO4J_PASSWORD already set'
}

Set-EnvIfBlank 'TEXT_MODEL'  'ai/gemma4'
Set-EnvIfBlank 'VLM_MODEL'   'ai/gemma4'
Set-EnvIfBlank 'EMBED_MODEL' 'ai/nomic-embed-text-v1.5'
$AppPort = ('' + (Get-EnvValue 'APP_PORT')).Trim()
if ($AppPort -notmatch '^[0-9]+$') { $AppPort = '8080' }
if ("$AppPort" -eq '') { $AppPort = '8080' }
$TextModel  = Get-EnvValue 'TEXT_MODEL'
$VlmModel   = Get-EnvValue 'VLM_MODEL'
$EmbedModel = Get-EnvValue 'EMBED_MODEL'
Ok ("models: {0} (text/vision), {1} (embeddings)" -f $TextModel, $EmbedModel)

# --- ports ----------------------------------------------------------------------
# Warn about occupied ports only when the stack is not already the occupant.
function Test-PortBusy([int]$Port) {
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        try {
            $c = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop
            if ($c) { return $true }
        } catch { return $false }
        return $false
    }
    # Builds without Get-NetTCPConnection still get a real probe: if a quick
    # connect to 127.0.0.1 succeeds, something is listening there.
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($async.AsyncWaitHandle.WaitOne(200) -and $client.Connected) { return $true }
        return $false
    } catch { return $false }
    finally { $client.Close() }
}

$runningNow = @()
try {
    $runningNow = @(Compose ps --status running --format '{{.Service}}' 2>$null |
        Where-Object { ('' + $_).Trim() -ne '' })
} catch { }
if ($runningNow.Count -gt 0) {
    Ok 'this stack is already running; leaving its ports alone'
} else {
    foreach ($p in @($AppPort, 7474, 7687)) {
        if (Test-PortBusy ([int]$p)) {
            Warn "port $p is already in use by something else - change it in .env if startup fails"
        }
    }
}

# --- Docker Model Runner ------------------------------------------------------------
# The models execute on the host through Model Runner, with the GPU - never
# inside the Docker VM. So there is no VM memory requirement beyond the app and
# the database, and no model service in docker-compose.yml at all.
Step 'Checking Docker Model Runner'
docker model version *> $null
if ($LASTEXITCODE -ne 0) {
    Fail ("This Docker installation has no Model Runner.`n" +
          '    It ships with Docker Desktop 4.40 or newer - update Docker Desktop and run start.ps1 again.')
}
docker model status *> $null
if ($LASTEXITCODE -ne 0) {
    Warn 'Model Runner is off; turning it on'
    docker desktop enable model-runner *> $null
    $waited = 0
    while ($true) {
        docker model status *> $null
        if ($LASTEXITCODE -eq 0) { break }
        if ($waited -ge 60) {
            Fail ("Could not enable Model Runner.`n" +
                  '    Turn it on in Docker Desktop > Settings > AI, then run start.ps1 again.')
        }
        Start-Sleep -Seconds 3
        $waited += 3
    }
}
Ok "Model Runner is running (models execute on this machine's GPU)"

Step 'Checking resources'
$freeGB = -1
try {
    $drive = (Get-Item $Root).PSDrive
    if ($null -ne $drive -and $null -ne $drive.Free) {
        $freeGB = [int][math]::Floor($drive.Free / 1GB)
    }
} catch { }
if ($freeGB -lt 0) {
    Warn 'could not determine free disk space; the models need about 8GB'
} elseif ($freeGB -lt 20) {
    Warn "only ${freeGB}GB free on this disk; the models need about 8GB"
} else {
    Ok "${freeGB}GB free disk"
}

# --- build and start -------------------------------------------------------------------
Step 'Building images (first run takes a few minutes)'
Compose build
if ($LASTEXITCODE -ne 0) {
    Fail 'The image build failed. Read the output above, fix the cause, then run start.ps1 again.'
}

Step 'Starting services'
Compose up -d
if ($LASTEXITCODE -ne 0) {
    Fail ("Could not start the services. See what they say:  {0} logs" -f $ComposeCmd)
}

function Wait-Healthy([string]$Name, [int]$Limit) {
    Write-Host -NoNewline "    waiting for $Name"
    $waited = 0
    while ($true) {
        $psOut = ''
        try { $psOut = (Compose ps --format json $Name 2>$null | Out-String) } catch { }
        if ($psOut -match '"Health"\s*:\s*"healthy"') { break }
        if ($waited -ge $Limit) {
            Write-Host ''
            Fail ("{0} did not become healthy in {1}s.`n    See what it says:  {2} logs {0}" -f $Name, $Limit, $ComposeCmd)
        }
        Write-Host -NoNewline '.'
        Start-Sleep -Seconds 3
        $waited += 3
    }
    Write-Host ' ok' -ForegroundColor Green
}
Wait-Healthy 'neo4j' 240
Wait-Healthy 'app' 180

# --- models -----------------------------------------------------------------------------
Step 'Downloading models (one time, about 8GB)'
$haveModels = @()
foreach ($line in @(docker model list 2>$null)) {
    $tok = (('' + $line) -split '\s+')[0]
    if (($tok -ne '') -and ($tok -ne 'MODEL')) { $haveModels += $tok }
}
$pulled = @()
foreach ($m in @($TextModel, $VlmModel, $EmbedModel)) {
    if ("$m" -eq '') { continue }
    if ($pulled -contains $m) { continue }
    $pulled += $m
    $short = $m -replace '^docker\.io/', ''
    if ($haveModels -contains $short) {
        Ok "$m is already here"
    } else {
        docker model pull $m
        if ($LASTEXITCODE -ne 0) {
            Fail "Could not pull $m. Check your connection and run start.ps1 again."
        }
    }
}
Ok ('models ready: ' + ($pulled -join ' '))

# --- prove it actually answers -------------------------------------------------------------
Step 'Checking the model answers (loads it into the GPU on first use)'
$proofOk = $false
try {
    $answer = (docker model run $TextModel 'Say OK.' 2>$null | Out-String).Trim()
    if (($LASTEXITCODE -eq 0) -and ($answer.Length -gt 0)) { $proofOk = $true }
} catch { }
if ($proofOk) {
    Ok 'the model loaded and answered'
} else {
    Warn 'The model did not answer a test prompt. The app will still open, but'
    Warn 'processing will fail. See:  docker model status  and  docker model logs'
}

# On an AMD card, Model Runner should be on the Vulkan llama.cpp backend.
# Surface what it reports so a silent CPU fallback does not masquerade as
# "working, just slow".
$statusText = ''
try { $statusText = (docker model status 2>&1 | Out-String) } catch { }
$backendLine = ''
foreach ($line in ($statusText -split "`r?`n")) {
    if (('' + $line) -match '(?i)backend|vulkan|cuda|gpu|llama') { $backendLine = ('' + $line).Trim(); break }
}
if ($backendLine -ne '') { Ok "model runner reports: $backendLine" }
if (-not ($statusText -match '(?i)vulkan|cuda|gpu')) {
    Warn "docker model status shows no GPU backend. If generation above felt slow, it is"
    Warn 'likely running on the CPU. Vulkan support for AMD cards ships with current Docker'
    Warn 'Desktop - update Docker Desktop (some Desktop upgrades have shipped without the'
    Warn 'GPU backend, so re-check after updating).'
}

# --- done -----------------------------------------------------------------------------------
$URL = "http://127.0.0.1:$AppPort"
Step 'Ready'
Write-Host ''
Write-Host "    $URL" -ForegroundColor Cyan
Write-Host ''
Write-Host '    Drop documents on the page and they process automatically.'
Write-Host ("    Stop with:      {0} down" -f $ComposeCmd)
Write-Host ("    See progress:   {0} logs -f app" -f $ComposeCmd)
Write-Host ''

# Keep this deployment current: a background watcher pulls any push to the
# repo and restarts the running containers on the new version. Skip with
# -NoAutoUpdate, stop later with:  .\auto-update.ps1 Stop
if (-not $NoAutoUpdate) {
    $au = Join-Path $Root 'auto-update.ps1'
    if ((Test-Path (Join-Path $Root '.git')) -and (Test-Path $au)) {
        try { & $au Start | ForEach-Object { Write-Host "    $_" } } catch { }
    }
}

if (-not $NoBrowser) {
    try { Start-Process $URL } catch { }
}
