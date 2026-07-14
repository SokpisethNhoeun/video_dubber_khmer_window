param(
    [string]$PublisherName = "Khmer Video Dubber",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "signing-output"),
    [Parameter(Mandatory = $true)]
    [SecureString]$Password
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null

$Certificate = New-SelfSignedCertificate `
    -Type CodeSigningCert `
    -Subject "CN=$PublisherName" `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -HashAlgorithm SHA256 `
    -KeyAlgorithm RSA `
    -KeyLength 3072 `
    -KeyExportPolicy Exportable `
    -NotAfter (Get-Date).AddYears(3)

$PfxPath = Join-Path $OutputDirectory "KhmerVideoDubber-Signing.pfx"
$CerPath = Join-Path $OutputDirectory "KhmerVideoDubber-Publisher.cer"
$PfxBase64Path = Join-Path $OutputDirectory "WINDOWS_SIGNING_PFX_BASE64.txt"

Export-PfxCertificate -Cert $Certificate -FilePath $PfxPath -Password $Password | Out-Null
Export-Certificate -Cert $Certificate -FilePath $CerPath -Type CERT | Out-Null
[IO.File]::WriteAllText($PfxBase64Path, [Convert]::ToBase64String([IO.File]::ReadAllBytes($PfxPath)))

Write-Host "Created free self-signed code-signing files in $OutputDirectory"
Write-Host "Set WINDOWS_SIGNING_PFX_BASE64 from $PfxBase64Path"
Write-Host "Set WINDOWS_SIGNING_PASSWORD to the password supplied to this script."
Write-Host "Keep the PFX and its Base64 file private. Give only the CER file to administrators."
