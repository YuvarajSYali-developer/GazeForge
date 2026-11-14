# GazeForge GNM - Windows smoke test (PowerShell)
#   powershell -ExecutionPolicy Bypass -File .\tools\windows_smoke_test.ps1 [-Blender "C:\path\to\blender.exe"]
# Runs: (1) gnm_eyes.py for all 3 inputs as quick low-res previews, (2) opens the Timeline .blend and renders one
# frame, (3) renders one portrait. Writes everything to demo_run\windows\ and a log file. Takes ~5-15 minutes.
param([string]$Blender = "")
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$out = Join-Path $root "demo_run\windows"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out "smoke_test_log.txt"
"GazeForge GNM Windows smoke test  $(Get-Date)" | Out-File $log

if (-not $Blender) {
  $cands = @()
  $cmd = Get-Command blender -ErrorAction SilentlyContinue
  if ($cmd) { $cands += $cmd.Source }
  $cands += Get-ChildItem "C:\Program Files\Blender Foundation" -Recurse -Filter blender.exe -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | ForEach-Object { $_.FullName }
  $cands += Get-ChildItem "$env:ProgramFiles\Steam\steamapps\common\Blender" -Filter blender.exe -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
  if ($cands.Count -gt 0) { $Blender = $cands[0] }
}
if (-not $Blender -or -not (Test-Path $Blender)) {
  Write-Host "Blender not found. Install Blender 4.2 LTS or newer, or pass -Blender C:\path\to\blender.exe" -ForegroundColor Red
  "FAIL: blender not found" | Out-File $log -Append; exit 1
}
& $Blender --version | Select-Object -First 1 | Tee-Object -FilePath $log -Append
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ff) { "NOTE: ffmpeg not on PATH - videos stay as PNG frames (install ffmpeg or 'pip install imageio-ffmpeg')" | Tee-Object -FilePath $log -Append }

$results = @()
function Run-Step($name, $argList, $check) {
  Write-Host "== $name" -ForegroundColor Cyan
  "== $name" | Out-File $log -Append
  & $Blender @argList 2>&1 | Select-String -Pattern "gazeforge|stills|SMOKE|Error|Traceback" | ForEach-Object { $_.Line } | Tee-Object -FilePath $log -Append
  $ok = & $check
  $script:results += [pscustomobject]@{Step=$name; Result=$(if ($ok) {"PASS"} else {"FAIL"})}
}

foreach ($inp in @("Left", "Right", "Generate Motion")) {
  $tag = ($inp -replace " ", "_").ToLower()
  $mp4 = Join-Path $out "$tag.mp4"
  Run-Step "gnm_eyes.py --input $inp" @("-b", "GazeForge_GNM.blend", "--python", "gnm_eyes.py", "--", "--input", $inp,
    "--res", "480x270", "--samples", "4", "--frame-step", "12", "--keep-frames", "--out", $mp4) `
    { (Test-Path $mp4) -or ((Get-ChildItem (Join-Path $out "${tag}_frames") -Filter *.png -ErrorAction SilentlyContinue).Count -gt 0) }
}
$tl = Join-Path $out "timeline_frame86.png"
$expr = "import bpy; sc=bpy.context.scene; print('SMOKE timeline frames', sc.frame_start, sc.frame_end, 'markers', len(sc.timeline_markers)); sc.cycles.samples=4; sc.render.resolution_x=480; sc.render.resolution_y=270; sc.frame_set(86); sc.render.filepath=r'$tl'; bpy.ops.render.render(write_still=True)"
Run-Step "Timeline .blend opens and renders" @("-b", "GazeForge_GNM_Timeline.blend", "--python-expr", $expr) { Test-Path $tl }
Run-Step "render_stills.py (set 05)" @("-b", "--python", "render_stills.py", "--", "--sets", "05", "--samples", "4",
  "--res", "432x540", "--out", $out) { Test-Path (Join-Path $out "set05_amara.png") }

$results | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $log -Append
if ($results.Result -contains "FAIL") { Write-Host "SOME STEPS FAILED - send demo_run\windows\smoke_test_log.txt" -ForegroundColor Red; exit 1 }
Write-Host "ALL STEPS PASSED  (outputs in demo_run\windows)" -ForegroundColor Green
