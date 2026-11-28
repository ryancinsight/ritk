param([Parameter(Mandatory)][string]$Executable)
$ErrorActionPreference = 'Stop'
# Cargo refreshes the exported EXE from its linker output even on a fresh
# build. MSVC preserves the basename to keep the embedded PDB path stable.
$export = Get-Item -LiteralPath $Executable
$linkerOutput = Get-Item -LiteralPath (Join-Path $export.DirectoryName "deps/$($export.Name)")
if ((Get-FileHash -LiteralPath $linkerOutput.FullName).Hash -ne (Get-FileHash -LiteralPath $export.FullName).Hash) {
    throw "MSVC linker output differs from '$Executable'."
}
$linkerOutput.FullName
