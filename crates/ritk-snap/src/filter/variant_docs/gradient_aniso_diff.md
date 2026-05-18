Gradient anisotropic diffusion filter (ITK `GradientAnisotropicDiffusionImageFilter`).

Reduces noise while preserving edges using the 6-neighbour direct-flux form:
`I_new(p) = I(p) + Δt · Σ_{q ∈ N₆(p)} c(|I(q)−I(p)|) · (I(q)−I(p))`
with `c(s) = exp(−(s/K)²)`.

Distinct from the Perona-Malik filter: conductance is applied to raw
intensity differences (not spacing-normalised gradients).

# Stability constraint
`time_step ≤ 1/6 ≈ 0.1667`. ITK default: 0.125.
