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
# A pull is always followed by a rebuild and a start, whether or not the stack
# was up. It used to rebuild only what it found already running, which meant a
# machine that had been stopped for any reason came back on whatever image was
# last built rather than on the code that had just been pulled - and said
# nothing, because from the outside a stale container and a current one look
# identical. Bringing a stopped stack up is the deliberate part of that: an
# updater that leaves the new code unbuilt has not finished the update.
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

$Interval = 10
if ("$Env:UPDATE_INTERVAL" -match '^\d+$') { $Interval = [int]$Env:UPDATE_INTERVAL }
$PidFile = Join-Path $Root '.auto-update.pid'
$LogFile = Join-Path $Root 'auto-update.log'

# Appending to the log is best-effort and must never be fatal. Windows takes an
# exclusive write handle, so a second copy of this script - a manual Once while
# the watcher is up, or a tail held open in another window - makes Add-Content
# throw. That had two costs, and the second was the expensive one: an
# unguarded Add-Content at the end of a pipeline leaves ITS failure in
# $LASTEXITCODE, so a rebuild that docker completed perfectly was reported as
# "rebuild FAILED". A few short retries cover the moment a rival writer holds
# the handle; losing a log line is a worse outcome than losing nothing, and a
# better one than lying about the build.
function Write-Log($Lines) {
    if ($null -eq $Lines) { return }
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try { Add-Content -Path $LogFile -Value $Lines -Encoding ASCII; return }
        catch { Start-Sleep -Milliseconds 150 }
    }
}

function Say([string]$Msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Msg
    Write-Output $line
    Write-Log $line
}

# The bash that can run scripts/pull-models.sh, or '' if there is none.
#
# Not `Get-Command bash`. On a default Windows install that finds
# C:\Windows\System32\bash.exe, which is the WSL launcher - and on a machine
# with no distro installed it answers "execvpe(/bin/bash) failed: No such file
# or directory" and exits 1, which this script then reported as a model fetch
# that went wrong. The one we want ships beside git, which is already a hard
# requirement here, so it is looked for there first and by the same reasoning
# everywhere else before PATH is trusted at all.
function Get-BashPath {
    $candidates = @()
    $git = (Get-Command git -ErrorAction SilentlyContinue).Source
    if ($git) {
        # ...\Git\cmd\git.exe -> ...\Git\bin\bash.exe
        $candidates += (Join-Path (Split-Path (Split-Path $git -Parent) -Parent) 'bin\bash.exe')
    }
    $candidates += (Join-Path $env:ProgramFiles 'Git\bin\bash.exe')
    if (${env:ProgramFiles(x86)}) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Git\bin\bash.exe')
    }
    $candidates += (Join-Path $env:LOCALAPPDATA 'Programs\Git\bin\bash.exe')
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    # Last resort: whatever PATH has, unless it is the WSL launcher.
    $onPath = (Get-Command bash -ErrorAction SilentlyContinue).Source
    if ($onPath -and ($onPath -notlike "$env:SystemRoot\*")) { return $onPath }
    return ''
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
    # lines, not just any output. What this decides is only the wording: the
    # rebuild happens either way.
    $running = @()
    try {
        $running = @(docker compose ps --status running --format '{{.Service}}' 2>$null |
            Where-Object { ('' + $_).Trim() -ne '' })
    } catch { }
    $wasRunning = ($running.Count -gt 0)
    if ($wasRunning) {
        Say 'stack is running - rebuilding and restarting'
    } else {
        Say 'stack is not running - rebuilding and starting it on the new code'
    }
    # Models live on the host's Model Runner, not in the image, so a rebuild
    # does not fetch them. Never fatal here: this runs unattended and must not
    # leave a machine stopped because a registry was briefly unreachable - the
    # app starts and reports the missing model itself. The script is shell, so
    # it needs the bash that ships with Git for Windows; git is already a
    # requirement of this file, but bash is not always on PATH beside it, and a
    # missing interpreter is worth naming rather than reporting as a fetch that
    # went wrong.
    if (Test-Path (Join-Path $Root 'scripts/pull-models.sh')) {
        $bash = Get-BashPath
        if ($bash) {
            $global:LASTEXITCODE = 0
            $pullOut = & $bash ./scripts/pull-models.sh 2>&1 | ForEach-Object { '' + $_ }
            $pullCode = $LASTEXITCODE
            Write-Log $pullOut
            if ($pullCode -ne 0) {
                Say 'model fetch reported a problem - see auto-update.log'
            }
        } else {
            Say 'no Git bash found - skipping the model fetch; run .\start.ps1 if a model is missing'
        }
    }
    # --remove-orphans: a push that retires a service from the compose file
    # must also retire its running container, or it lingers forever.
    # The output is captured first and logged afterwards, so that the verdict
    # below reads docker's exit code and not the log writer's - see Write-Log.
    # A stale $LASTEXITCODE from an earlier command must not be read as this
    # rebuild's verdict if docker itself fails to launch.
    $global:LASTEXITCODE = 1
    $buildOut = docker compose up -d --build --remove-orphans 2>&1 | ForEach-Object { '' + $_ }
    $buildCode = $LASTEXITCODE
    Write-Log $buildOut
    if ($buildCode -eq 0) {
        Say 'running on the new version'
    } elseif ($wasRunning) {
        Say 'rebuild FAILED - the old containers may still be running; see auto-update.log'
    } else {
        Say 'rebuild FAILED - nothing is running; see auto-update.log'
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Say 'git is not on PATH - install Git for Windows: https://git-scm.com/download/win'
    exit 1
}

switch ($Verb) {
    'Once' {
        # Not refused, because a manual nudge is a reasonable thing to want
        # while the watcher is up - but said out loud, because the two race on
        # the same checkout and the same compose project, and the loser of that
        # race sees a pull that has already happened and a rebuild it did not
        # start. Without this line that reads as this run having done nothing.
        $alive = Get-WatcherPid
        if ($alive -ne 0) {
            Say "note: the background watcher is also running (pid $alive) - it may get there first"
        }
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
