param(
    [Parameter(Mandatory = $true)][string]$Signtool,
    [Parameter(Mandatory = $true)][string]$Pfx,
    [Parameter(Mandatory = $true)][string]$Password,
    [Parameter(Mandatory = $true)][string]$FilePath,
    [string]$TimestampUrl = "",
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$StartInfo = [Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $Signtool
$StartInfo.UseShellExecute = $false
$StartInfo.RedirectStandardOutput = $true
$StartInfo.RedirectStandardError = $true

$Arguments = @("sign", "/fd", "SHA256", "/f", $Pfx, "/p", $Password)
if ($TimestampUrl) {
    $Arguments += @("/td", "SHA256", "/tr", $TimestampUrl)
}
$Arguments += $FilePath
foreach ($Argument in $Arguments) {
    $StartInfo.ArgumentList.Add($Argument)
}

$Process = [Diagnostics.Process]::new()
$Process.StartInfo = $StartInfo
try {
    if (-not $Process.Start()) {
        throw "Could not start signtool for $FilePath."
    }
    if (-not $Process.WaitForExit($TimeoutSeconds * 1000)) {
        $Process.Kill($true)
        throw "Signing timed out after $TimeoutSeconds seconds for $FilePath."
    }
    $StandardOutput = $Process.StandardOutput.ReadToEnd()
    $StandardError = $Process.StandardError.ReadToEnd()
    if ($StandardOutput) { Write-Host $StandardOutput.TrimEnd() }
    if ($StandardError) { Write-Host $StandardError.TrimEnd() }
    if ($Process.ExitCode -ne 0) {
        throw "Signtool failed with exit code $($Process.ExitCode) for $FilePath."
    }
}
finally {
    $Process.Dispose()
}
