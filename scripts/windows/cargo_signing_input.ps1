param([Parameter(Mandatory)][string]$Executable)
$ErrorActionPreference = 'Stop'
# Cargo refreshes the exported EXE from its linker output even on a fresh
# build. A signer can replace a file and break Cargo's original hard link.
$export = Get-Item -LiteralPath $Executable
$hash = (Get-FileHash -LiteralPath $export.FullName).Hash
$deps = Join-Path $export.DirectoryName 'deps'
$candidates = @(Get-ChildItem -LiteralPath $deps -File -Filter "$($export.BaseName)-*.exe" |
    Where-Object { $_.Length -eq $export.Length -and (Get-FileHash -LiteralPath $_.FullName).Hash -eq $hash })
if ($candidates.Count -ne 1) {
    throw "Expected one linker output matching '$Executable', found $($candidates.Count)."
}
$candidates[0].FullName
