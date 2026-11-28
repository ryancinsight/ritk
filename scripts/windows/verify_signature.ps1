param(
    [Parameter(Mandatory)][string]$Path,
    [Parameter(Mandatory)][string]$ExpectedSubject
)
$ErrorActionPreference = 'Stop'
$signature = Get-AuthenticodeSignature -LiteralPath $Path
if ($signature.Status -ne 'Valid' -or $signature.SignatureType -ne 'Authenticode') {
    throw "Invalid embedded Authenticode signature for '$Path' ($($signature.Status))."
}
if ($signature.SignerCertificate.Subject -cne $ExpectedSubject) {
    throw "Unexpected publisher for '$Path': $($signature.SignerCertificate.Subject)."
}
if (-not $signature.TimeStamperCertificate) {
    throw "Missing signature timestamp for '$Path'."
}
Write-Host "Verified $Path ($ExpectedSubject)"
