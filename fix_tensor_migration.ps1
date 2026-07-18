#!/usr/bin/env pwsh
# Migration script: burn Tensor<B,D> → coeus Tensor<T,B>
# Excludes: burn_compat_types.rs, types.rs, ritk-tensor-ops/src/lib.rs

$ErrorActionPreference = "Stop"
$root = "D:\atlas\repos\ritk"
$excludePatterns = @("burn_compat_types\.rs", "ritk-image\\src\\types\.rs", "ritk-tensor-ops\\src\\lib\.rs")

$files = Get-ChildItem -Path "$root\crates" -Recurse -Filter "*.rs" | Where-Object {
    $path = $_.FullName
    $excluded = $false
    foreach ($pat in $excludePatterns) {
        if ($path -match $pat) { $excluded = $true; break }
    }
    -not $excluded
}

$modified = 0
foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    $original = $content

    # ── 1. Tensor type param fixes ──────────────────────────────────────────
    # Order matters: fix 3-param patterns BEFORE 2-param

    # 1a. Turbofish 3-param with Int: Tensor::<B, N, ritk_image::tensor::Int>
    $content = $content -replace 'Tensor::<([A-Za-z_]\w*),\s*(\d+|D),\s*ritk_image::tensor::Int>', 'Tensor::<i32, $1>'

    # 1b. Turbofish 3-param with D,D: Tensor::<B, D, D>
    $content = $content -replace 'Tensor::<([A-Za-z_]\w*),\s*D,\s*D>', 'Tensor::<f32, $1>'

    # 1c. Turbofish 2-param: Tensor::<B, N> where N is digit
    $content = $content -replace 'Tensor::<([A-Za-z_]\w*),\s*(\d+)>', 'Tensor::<f32, $1>'

    # 1d. Turbofish 2-param: Tensor::<B, D> where D is const generic
    $content = $content -replace 'Tensor::<([A-Za-z_]\w*),\s*D>', 'Tensor::<f32, $1>'

    # 1e. Type-position 2-param: Tensor<B, N> where N is digit
    $content = $content -replace 'Tensor<([A-Za-z_]\w*),\s*(\d+)>', 'Tensor<f32, $1>'

    # 1f. Type-position 2-param: Tensor<B, D> where D is const generic
    $content = $content -replace 'Tensor<([A-Za-z_]\w*),\s*D>', 'Tensor<f32, $1>'

    # ── 2. AutodiffBackend → Backend ─────────────────────────────────────────
    $content = $content -replace '\bAutodiffBackend\b', 'Backend'

    # ── 3. ritk_image::tensor::Int → coeus_core::Int ────────────────────────
    $content = $content -replace 'ritk_image::tensor::Int', 'coeus_core::Int'

    # ── 4. AutodiffModule → comment out ──────────────────────────────────────
    # Handle single-line use statements
    $content = $content -replace '^(use\s+ritk_image::burn::module::\{[^}]*AutodiffModule[^}]*\};)', '// TODO: migrate AutodiffModule - not available in coeus: /* $1 */'
    $content = $content -replace '^(use\s+ritk_image::burn::module::AutodiffModule;)', '// TODO: migrate AutodiffModule - not available in coeus: /* $1 */'

    # ── 5. ritk_image::burn imports → comment out ────────────────────────────
    # Handle various burn import patterns
    $content = $content -replace '^(use\s+ritk_image::burn::backend::Autodiff;)', '// TODO: migrate burn::backend::Autodiff - not available in coeus: /* $1 */'
    $content = $content -replace '^(use\s+ritk_image::burn::module::\{[^}]*\};)', '// TODO: migrate burn::module - not available in coeus: /* $1 */'
    $content = $content -replace '^(use\s+ritk_image::burn::module::\w+;)', '// TODO: migrate burn::module - not available in coeus: /* $1 */'
    $content = $content -replace '^(use\s+ritk_image::burn::optim[^;]*;)', '// TODO: migrate burn::optim - not available in coeus: /* $1 */'
    $content = $content -replace '^(use\s+ritk_image::burn::nn::Linear;)', '// TODO: migrate burn::nn::Linear - not available in coeus: /* $1 */'
    $content = $content -replace '^(use\s+ritk_image::burn::record[^;]*;)', '// TODO: migrate burn::record - not available in coeus: /* $1 */'
    $content = $content -replace '^(use\s+ritk_image::burn::\{[^}]*\};)', '// TODO: migrate burn - not available in coeus: /* $1 */'

    # ── 6. from_data(TensorData::new(...)) → from_slice_on(...) ─────────────
    # Tensor::<f32, B>::from_data(TensorData::new(data, Shape::new(dims)), &device)
    # → Tensor::<f32, B>::from_slice_on(dims, &data, &device)
    $content = $content -replace 'Tensor::<(\w+),\s*(\w+)>::from_data\(TensorData::new\((\w+),\s*Shape::new\(([^)]+)\)\),\s*&(\w+)\)', 'Tensor::<$1, $2>::from_slice_on($4, &$3, &$5)'
    # With device passed directly (not referenced)
    $content = $content -replace 'Tensor::<(\w+),\s*(\w+)>::from_data\(TensorData::new\((\w+),\s*Shape::new\(([^)]+)\)\),\s*(\w+)\)', 'Tensor::<$1, $2>::from_slice_on($4, &$3, $5)'

    # ── 7. Tensor::<B,N>::zeros(device) → Tensor::<f32,B>::zeros(shape) ─────
    # This is complex - zeros in coeus takes shape, not device
    # Leave for manual fixup

    # ── 8. Standalone TensorData usage cleanup ───────────────────────────────
    # Remove TensorData from imports where it appears with other items
    $content = $content -replace '(\bTensorData\b,?\s*)', ''
    $content = $content -replace '(\s*,\s*\bTensorData\b)', ''
    # Clean up empty braces from import removal: { , Foo} → { Foo }
    $content = $content -replace '\{\s*,\s*', '{ '
    $content = $content -replace ',\s*\}', ' }'
    # Clean up fully empty use statements
    $content = $content -replace 'use\s+ritk_image::tensor::\{\s*\};\s*\n', ''

    if ($content -ne $original) {
        Set-Content $file.FullName -Value $content -NoNewline -Encoding UTF8
        $modified++
        Write-Host "Modified: $($file.FullName | Split-Path -Leaf)"
    }
}

Write-Host "`n=== Modified $modified files ==="
