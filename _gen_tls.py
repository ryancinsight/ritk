
import pathlib

BT = chr(96)*3  # triple backtick

content = f"""//! Threshold Level Set segmentation for 3-D medical images.
//!
//! # Mathematical Specification
//!
//! Evolves a level set function phi according to:
//!
//! {BT}text
//!   d_phi/dt = |grad_phi| * (curvature_weight * kappa + propagation_weight * T(I(x)))
//! {BT}
//!
//! where:
//! - kappa = div(grad_phi / |grad_phi|) is the mean curvature.
//! - T(I(x)) is the threshold speed function:
//!   T(I) = +1.0 if lower_threshold <= I <= upper_threshold,
//!   T(I) = -1.0 otherwise.
//!
//! The discrete update uses:
//!   speed = curvature_weight * kappa[i] - propagation_weight * T[i]
//!   dphi  = dt * |grad_phi| * speed
//!
//! When T = +1 (inside threshold range), the negative propagation term causes
//! phi to decrease, expanding the contour. When T = -1 (outside range), the
//! positive propagation term causes phi to increase, contracting the contour.
//!
//! ## Discretisation
//!
//! Forward Euler time stepping. Spatial derivatives use second-order central
//! finite differences with clamped (Neumann) boundary conditions.
//!
//! ## Convergence
//!
//! Iteration terminates when:
//! - {chr(96)}max |dphi| / dt < tolerance{chr(96)}, or
//! - {chr(96)}iteration == max_iterations{chr(96)}.
//!
//! ## Output
//!
//! Binary mask: 1.0 where phi < 0 (inside), 0.0 elsewhere.
//!
//! # References
//!
//! - Sethian, J. A. (1999). *Level Set Methods and Fast Marching Methods*.
//!   Cambridge University Press.
"""

print(f'Generated {{len(content)}} chars')
