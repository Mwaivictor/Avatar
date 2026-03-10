# ─────────────────────────────────────────────────────────────────
# Install Virtual Camera & Microphone Drivers
# RIGHT-CLICK → "Run with PowerShell" as Administrator
# ─────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$downloads = "$env:TEMP\AvatarSetup"
New-Item -ItemType Directory -Path $downloads -Force | Out-Null

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Avatar Virtual Device Installer" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# ─── 1. OBS Studio (provides OBS Virtual Camera) ────────────────
$obsInstalled = Test-Path "C:\Program Files\obs-studio\bin\64bit\obs64.exe"
if ($obsInstalled) {
    Write-Host "[OK] OBS Studio already installed" -ForegroundColor Green
} else {
    Write-Host "[>>] Downloading OBS Studio..." -ForegroundColor Yellow
    $obsUrl = "https://cdn-fastly.obsproject.com/downloads/OBS-Studio-31.0.2-Windows-Installer.exe"
    $obsFile = "$downloads\OBS-Studio-Installer.exe"
    Invoke-WebRequest -Uri $obsUrl -OutFile $obsFile -UseBasicParsing
    Write-Host "[>>] Installing OBS Studio (silent)..." -ForegroundColor Yellow
    Start-Process -FilePath $obsFile -ArgumentList "/S" -Wait
    Write-Host "[OK] OBS Studio installed" -ForegroundColor Green
}

# ─── 2. VB-Audio Virtual Cable (provides CABLE Output mic) ──────
$vbInstalled = Get-PnpDevice -FriendlyName "*VB-Audio*","*CABLE*" -ErrorAction SilentlyContinue
if ($vbInstalled) {
    Write-Host "[OK] VB-Audio Virtual Cable already installed" -ForegroundColor Green
} else {
    Write-Host "[>>] Downloading VB-Audio Virtual Cable..." -ForegroundColor Yellow
    $vbUrl = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
    $vbZip = "$downloads\VBCABLE.zip"
    $vbDir = "$downloads\VBCABLE"
    Invoke-WebRequest -Uri $vbUrl -OutFile $vbZip -UseBasicParsing
    Expand-Archive -Path $vbZip -DestinationPath $vbDir -Force
    Write-Host "[>>] Installing VB-Audio Virtual Cable..." -ForegroundColor Yellow
    Write-Host "     (A driver install dialog may appear - click Install)" -ForegroundColor DarkYellow
    $setupExe = Get-ChildItem -Path $vbDir -Filter "VBCABLE_Setup_x64.exe" -Recurse | Select-Object -First 1
    if ($setupExe) {
        Start-Process -FilePath $setupExe.FullName -ArgumentList "-i" -Wait -Verb RunAs
    } else {
        $setupExe = Get-ChildItem -Path $vbDir -Filter "VBCABLE_Setup.exe" -Recurse | Select-Object -First 1
        Start-Process -FilePath $setupExe.FullName -ArgumentList "-i" -Wait -Verb RunAs
    }
    Write-Host "[OK] VB-Audio Virtual Cable installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host " Installation Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: You must RESTART your computer for the" -ForegroundColor Yellow
Write-Host "virtual devices to appear in all applications." -ForegroundColor Yellow
Write-Host ""
Write-Host "After reboot:" -ForegroundColor Cyan
Write-Host "  1. Run: python start.py" -ForegroundColor White
Write-Host "  2. Click 'Start Avatar' in the dashboard" -ForegroundColor White
Write-Host "  3. In Google Meet/Zoom/Teams settings:" -ForegroundColor White
Write-Host "     Camera → OBS Virtual Camera" -ForegroundColor White
Write-Host "     Microphone → CABLE Output (VB-Audio Virtual Cable)" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
