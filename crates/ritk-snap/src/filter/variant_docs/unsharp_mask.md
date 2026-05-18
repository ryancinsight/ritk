Unsharp mask sharpening filter (ITK `UnsharpMaskingImageFilter` parity).

Sharpens edges by adding back a scaled, thresholded version of the
high-frequency component `I − G_σ∗I` to the original image:
`output = I + amount · max(0, |I − blur(I)| − threshold) · sign(I − blur(I))`

# Invariants
- `amount = 0.0` → output equals input.
- uniform input → output equals input.
- `clamp = true` → output ∈ `[min(I), max(I)]`.
