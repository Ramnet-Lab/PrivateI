# Keep this deployment tracking the GitHub repo (Windows).
#
#   .\auto-update.ps1 Once      check now; pull and restart if there is a push
#   .\auto-update.ps1 Watch     keep checking in the foreground
#   .\auto-update.ps1 Start     keep checking in the background
#   .\auto-update.ps1 Stop      stop the background watcher
#   .\auto-update.ps1 Status    is the watcher running, and are we current?
#
# Pull-based on purpose: a webhook needs a public endpoint and a self-hosted
# GitHub runner means handing this machine a repo token and a remote-execution
# surface. Polling git gets the same outcome with neither.
#
# The containers are rebuilt and restarted ONLY when the stack is already
# running; a stopped stack just gets the new code and stays stopped.
#
# Written for Windows PowerShell 5.1, ASCII only. Every line is also appended
# to auto-update.log, because the hidden background watcher has no console -
# the log is the only place its output can go.

param(
    [Parameter(Position = 0)]
    [ValidateSet('Once', 'Watch', 'Start', 'Stop', 'Status')]
    [string]$Verb = 'Once'
)

$ErrorActionPreference = 'Continue'
$Root = $PSScriptRoot
Set-Location $Root

$Interval = 300
if ("$Env:UPDATE_INTERVAL" -match '^\d+$') { $Interval = [int]$Env:UPDATE_INTERVAL }
$PidFile = Join-Path $Root '.auto-update.pid'
$LogFile = Join-Path $Root 'auto-update.log'

function Say([string]$Msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Msg
    Write-Output $line
    try { Add-Content -Path $LogFile -Value $line -Encoding ASCII } catch { }
}

function Get-WatcherPid {
    # A pidfile is only trusted if the process it names is actually alive;
    # a stale file from a crash or reboot counts as "not running".
    if (Test-Path $PidFile) {
        $raw = ''
        try { $raw = ('' + (Get-Content $PidFile -TotalCount 1)).Trim() } catch { }
        if ($raw -match '^\d+$') {
            $proc = Get-Process -Id ([int]$raw) -ErrorAction SilentlyContinue
            if ($proc) { return [int]$raw }
        }
    }
    return 0
}

function Invoke-CheckOnce {
    # Never touch a checkout that has local edits.
    git diff --quiet 2>$null
    $dirty = ($LASTEXITCODE -ne 0)
    git diff --cached --quiet 2>$null
    if ($LASTEXITCODE -ne 0) { $dirty = $true }
    if ($dirty) {
        Say 'local changes present - not touching this checkout'
        return
    }

    git fetch -q origin 2>$null
    if ($LASTEXITCODE -ne 0) {
        Say 'fetch failed (offline?); will try again'
        return
    }

    $here = ('' + (git rev-parse HEAD 2>$null)).Trim()
    $there = ('' + (git rev-parse origin/main 2>$null)).Trim()
    if (($here -eq '') -or ($there -eq '')) {
        Say 'could not read git revisions - resolve by hand'
        return
    }
    if ($here -eq $there) { return }

    Say ('update found: {0} -> {1}' -f $here.Substring(0, 7), $there.Substring(0, 7))
    git merge-base --is-ancestor HEAD origin/main 2>$null
    if ($LASTEXITCODE -ne 0) {
        Say 'local history has diverged from origin/main - resolve by hand'
        return
    }

    git pull -q --ff-only origin main 2>$null
    if ($LASTEXITCODE -ne 0) {
        Say 'pull failed'
        return
    }
    $last = ('' + (git log -1 --format='%h %s' 2>$null)).Trim()
    if ($last.Length -gt 70) { $last = $last.Substring(0, 70) }
    Say "pulled $last"

    # ps prints a header row even with nothing running, so count real service
    # lines, not just any output.
    $running = @()
    try {
        $running = @(docker compose ps --status running --format '{{.Service}}' 2>$null |
            Where-Object { ('' + $_).Trim() -ne '' })
    } catch { }
    if ($running.Count -gt 0) {
        Say 'stack is running - rebuilding and restarting'
        # --remove-orphans: a push that retires a service from the compose file
        # must also retire its running container, or it lingers forever.
        # A stale $LASTEXITCODE from an earlier command must not be read as
        # this rebuild's verdict if docker itself fails to launch.
        $global:LASTEXITCODE = 1
        docker compose up -d --build --remove-orphans 2>&1 |
            ForEach-Object { '' + $_ } |
            Add-Content -Path $LogFile -Encoding ASCII
        if ($LASTEXITCODE -eq 0) {
            Say 'restarted on the new version'
        } else {
            Say 'rebuild FAILED - the old containers may still be running; see auto-update.log'
        }
    } else {
        Say 'stack is not running - code updated, nothing restarted'
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Say 'git is not on PATH - install Git for Windows: https://git-scm.com/download/win'
    exit 1
}

switch ($Verb) {
    'Once' {
        Invoke-CheckOnce
    }
    'Watch' {
        Say "watching origin/main every ${Interval}s (ctrl-c to stop)"
        while ($true) {
            Invoke-CheckOnce
            Start-Sleep -Seconds $Interval
        }
    }
    'Start' {
        $alive = Get-WatcherPid
        if ($alive -ne 0) {
            Say "already running (pid $alive)"
            exit 0
        }
        $watcher = Start-Process -FilePath 'powershell' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $PSCommandPath), 'Watch') -WindowStyle Hidden -PassThru -ErrorAction SilentlyContinue
        if ($null -eq $watcher) {
            Say 'could not start the background watcher - run .\auto-update.ps1 Watch in a window instead'
            exit 1
        }
        Set-Content -Path $PidFile -Value $watcher.Id -Encoding ASCII
        Say "watcher started (pid $($watcher.Id), every ${Interval}s, log: auto-update.log)"
    }
    'Stop' {
        $alive = Get-WatcherPid
        if ($alive -ne 0) {
            Stop-Process -Id $alive -Force -ErrorAction SilentlyContinue
            Remove-Item $PidFile -ErrorAction SilentlyContinue
            Say 'watcher stopped'
        } else {
            Remove-Item $PidFile -ErrorAction SilentlyContinue
            Say 'watcher was not running'
        }
    }
    'Status' {
        $alive = Get-WatcherPid
        if ($alive -ne 0) {
            Say "watcher running (pid $alive)"
        } else {
            Say 'watcher not running'
        }
        git fetch -q origin 2>$null
        $here = ('' + (git rev-parse HEAD 2>$null)).Trim()
        $there = ('' + (git rev-parse origin/main 2>$null)).Trim()
        if (($here -ne '') -and ($here -eq $there)) {
            Say 'checkout is current with origin/main'
        } else {
            Say 'an update is available - run: .\auto-update.ps1 Once'
        }
    }
}
