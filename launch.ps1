# The laptop's detaching layer for loop.py (badapples, step 7; LOG 2026-09-26).
#
# WSL2 stops its VM once no wsl.exe session references it, so a driver detached inside
# WSL would die with the session that started it. This script is the Windows-side anchor:
# it holds one wsl.exe call in the foreground for the whole run, keeps the machine awake
# (SetThreadExecutionState, the caffeinate -is equivalent, no admin needed) and appends
# the driver's output to <run>/driver.log. Run it from a user-level scheduled task
# (schtasks /run /tn badapples_core) so it survives any console; the same command after
# a crash resumes from the last completed stage (loop.py run resumes).
# MLX_ENABLE_CACHE_THRASHING_CHECK=0 (user's decision, LOG 2026-09-26 step 7): the CUDA backend's
# graph LRU cache throws once its cumulative misses pass twice its capacity; decode steps miss on
# nearly every step here, so the check is a performance diagnostic that would stop every stage.
param([Parameter(Mandatory = $true)][string]$Run)   # e.g. runs/core
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$wslRepo = '/mnt/c/Users/alin/Desktop/badapples'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
# ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x1) | ES_AWAYMODE_REQUIRED (0x40) = 2147483713; a hex literal that size is a negative Int32 in PowerShell 5.1
[Win32.Power]::SetThreadExecutionState([uint32]2147483713) | Out-Null
$runDir = Join-Path $repo $Run
$log = Join-Path $runDir 'driver.log'
Set-Content -Path (Join-Path $runDir 'driver.pid') -Value $PID
Add-Content -Path $log -Value "### $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') launch.ps1 pid $PID run $Run"
& cmd.exe /c "wsl.exe -d Ubuntu-24.04 --cd $wslRepo -- env HF_HOME=/home/alin/hf HF_HUB_OFFLINE=1 MLX_ENABLE_CACHE_THRASHING_CHECK=0 /home/alin/venvs/mlxcuda/bin/python loop.py run --run $Run >> $log 2>&1"
exit $LASTEXITCODE
