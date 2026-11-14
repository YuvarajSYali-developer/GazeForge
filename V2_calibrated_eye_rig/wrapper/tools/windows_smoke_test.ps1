<#
GazeForge - Windows smoke test with a real Blender executable (PowerShell 5+).

    powershell -ExecutionPolicy Bypass -File wrapper\tools\windows_smoke_test.ps1 [-Blender "C:\path\to\blender.exe"]

Run from anywhere; paths are resolved relative to this script. Writes ONLY to demo_run\windows\
(never to renders\, metadata.json or reports\). Takes a few minutes at 4 samples.
Steps:
  1. find blender.exe (argument, PATH, or C:\Program Files\Blender Foundation\Blender *\)
  2. blender --version
  3. one pose through the public API:  eyes.py --move 1 0  -> right.png
  4. replay all 10 poses of metadata.json + 5-direction test (wrapper\tools\cross_version_check.py)
  5. writes demo_run\windows\windows_smoke_test.json and prints PASS/FAIL
#>
param([string]$Blender = "", [int]$Samples = 4)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Out = Join-Path $Root "demo_run\windows"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

if (-not $Blender) {
    $cmd = Get-Command blender -ErrorAction SilentlyContinue
    if ($cmd) { $Blender = $cmd.Source }
    else {
        $cand = Get-ChildItem "C:\Program Files\Blender Foundation\Blender *\blender.exe" -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending | Select-Object -First 1
        if ($cand) { $Blender = $cand.FullName }
    }
}
if (-not $Blender -or -not (Test-Path $Blender)) { Write-Error "blender.exe not found - pass -Blender <path>" }
Write-Host "Blender: $Blender"
$ver = (& $Blender --version | Select-Object -First 1)
Write-Host $ver

$sw = [Diagnostics.Stopwatch]::StartNew()
$scene = Join-Path $Root "scene.blend"
$eyesPy = Join-Path $Root "wrapper\eyes.py"
$png = Join-Path $Out "right.png"
& $Blender -b $scene --python $eyesPy -- --move 1 0 --samples $Samples --out $png | Out-File (Join-Path $Out "api_run.log") -Encoding utf8
$apiOk = (Test-Path $png)
$t1 = $sw.Elapsed.TotalSeconds

$xv = Join-Path $Root "wrapper\tools\cross_version_check.py"
$lines = & $Blender -b --factory-startup --python $xv -- --samples $Samples --out (Join-Path $Out "direction_test")
$lines | Out-File (Join-Path $Out "cross_version_run.log") -Encoding utf8
$cv = ($lines | Select-String -Pattern "^CROSSVER " | Select-Object -Last 1)
$t2 = $sw.Elapsed.TotalSeconds
if (-not $cv) { Write-Error "cross_version_check produced no result - see demo_run\windows\cross_version_run.log" }
$res = ($cv.ToString().Substring(9) | ConvertFrom-Json)
$pass = $apiOk -and ($res.direction_validation -eq "5/5") -and ($res.replay_max_pupil2D_diff_px -lt 1e-6)
$report = [ordered]@{
    os = [Environment]::OSVersion.VersionString; blender = $ver; blender_exe = (Split-Path -Leaf $Blender)
    api_move_render_ok = $apiOk; api_seconds = [math]::Round($t1, 1)
    replay_max_pupil2D_diff_px = $res.replay_max_pupil2D_diff_px; replay_max_rotation_diff_rad = $res.replay_max_rotation_diff_rad
    direction_validation = $res.direction_validation; tests = $res.tests; samples = $Samples
    total_seconds = [math]::Round($t2, 1); result = $(if ($pass) { "PASS" } else { "FAIL" })
}
$report | ConvertTo-Json -Depth 4 | Out-File (Join-Path $Out "windows_smoke_test.json") -Encoding utf8
$report | ConvertTo-Json -Depth 4
Write-Host ("WINDOWS SMOKE TEST: " + $report.result)
