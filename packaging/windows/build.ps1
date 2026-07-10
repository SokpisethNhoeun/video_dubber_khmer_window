param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^https://')]
    [string]$LicenseServerUrl,
    [string]$Version = "1.0.0",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Generated = Join-Path $PSScriptRoot "generated"
$VendorBin = Join-Path $PSScriptRoot "vendor\bin"
$Dist = Join-Path $Root "dist\KhmerVideoDubber"

foreach ($Name in @("ffmpeg.exe", "ffprobe.exe")) {
    $Source = Join-Path $VendorBin $Name
    if (-not (Test-Path $Source)) {
        throw "Missing $Source. Place trusted Windows FFmpeg binaries in packaging\windows\vendor\bin."
    }
}

$ChecksumFile = Join-Path $VendorBin "SHA256SUMS.txt"
$ChecksumText = Get-Content $ChecksumFile -Raw
foreach ($Name in @("ffmpeg.exe", "ffprobe.exe")) {
    $Pattern = "(?im)^([a-f0-9]{64})\s+\*?" + [regex]::Escape($Name) + '$'
    $Match = [regex]::Match($ChecksumText, $Pattern)
    if (-not $Match.Success) {
        throw "Missing verified SHA-256 entry for $Name in $ChecksumFile."
    }
    $Actual = (Get-FileHash (Join-Path $VendorBin $Name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Match.Groups[1].Value.ToLowerInvariant()) {
        throw "SHA-256 verification failed for $Name."
    }
}

New-Item -ItemType Directory -Force (Join-Path $Generated "bin") | Out-Null
Copy-Item (Join-Path $VendorBin "ffmpeg.exe") (Join-Path $Generated "bin\ffmpeg.exe") -Force
Copy-Item (Join-Path $VendorBin "ffprobe.exe") (Join-Path $Generated "bin\ffprobe.exe") -Force

$ReleaseConfig = @{
    LICENSE_SERVER_URL = $LicenseServerUrl.TrimEnd('/')
    TRANSCRIPT_REVIEW_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    TRANSCRIPT_REVIEW_MODEL = "gemini-3.1-flash-lite"
} | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $Generated "release.json"), $ReleaseConfig, [Text.UTF8Encoding]::new($false))

python -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot "KhmerVideoDubber.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

if (-not (Test-Path (Join-Path $Dist "KhmerVideoDubber.exe"))) {
    throw "Build completed without KhmerVideoDubber.exe."
}

if (-not $SkipInstaller) {
    $Iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $Iscc) { throw "Inno Setup ISCC.exe is not in PATH." }
    & $Iscc.Source "/DMyAppVersion=$Version" (Join-Path $PSScriptRoot "KhmerVideoDubber.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed." }
}

Write-Host "Windows release build completed for version $Version."
