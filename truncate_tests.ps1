$files = @(
    @{ path = 'crates/ritk-transform/src/transform/bspline/mod.rs'; keepLines = 137 },
    @{ path = 'crates/ritk-transform/src/transform/affine/affine.rs'; keepLines = 0 },
    @{ path = 'crates/ritk-transform/src/transform/affine/rigid.rs'; keepLines = 0 },
    @{ path = 'crates/ritk-registration/src/optimizer/trait_.rs'; keepLines = 0 }
)

# For mod.rs: keep 137 lines (0..136)
$f = 'crates/ritk-transform/src/transform/bspline/mod.rs'
$l = Get-Content $f
Set-Content $f $l[0..136]
Write-Host "Truncated $f to 137 lines"

# For affine.rs: find line with '#[cfg(test)]' followed by '#[path = "tests_affine.rs"]'
$f = 'crates/ritk-transform/src/transform/affine/affine.rs'
$l = Get-Content $f
$idx = -1
for ($i = 0; $i -lt $l.Count - 1; $i++) {
    if ($l[$i] -eq '#[cfg(test)]' -and $l[$i+1] -eq '#[path = "tests_affine.rs"]') {
        $idx = $i + 2  # index of 'mod tests;'
        break
    }
}
if ($idx -ge 0) {
    Set-Content $f $l[0..$idx]
    Write-Host "Truncated $f at line $($idx+1)"
} else {
    Write-Host "Pattern not found in $f"
}

# For rigid.rs
$f = 'crates/ritk-transform/src/transform/affine/rigid.rs'
$l = Get-Content $f
$idx = -1
for ($i = 0; $i -lt $l.Count - 1; $i++) {
    if ($l[$i] -eq '#[cfg(test)]' -and $l[$i+1] -eq '#[path = "tests_rigid.rs"]') {
        $idx = $i + 2
        break
    }
}
if ($idx -ge 0) {
    Set-Content $f $l[0..$idx]
    Write-Host "Truncated $f at line $($idx+1)"
} else {
    Write-Host "Pattern not found in $f"
}

# For trait_.rs
$f = 'crates/ritk-registration/src/optimizer/trait_.rs'
$l = Get-Content $f
$idx = -1
for ($i = 0; $i -lt $l.Count - 1; $i++) {
    if ($l[$i] -eq '#[cfg(test)]' -and $l[$i+1] -eq '#[path = "tests_trait.rs"]') {
        $idx = $i + 2
        break
    }
}
if ($idx -ge 0) {
    Set-Content $f $l[0..$idx]
    Write-Host "Truncated $f at line $($idx+1)"
} else {
    Write-Host "Pattern not found in $f"
}
