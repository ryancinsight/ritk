//! D = 2: Zhangâ€“Suen thinning.
//!
//! Two sub-iterations per pass on 8-connected foreground / 4-connected
//! background. Neighbors Pâ‚‚..Pâ‚‰ are labeled clockwise from north.

/// Read a pixel from the mask, treating out-of-bounds as background.
#[inline]
fn pixel(mask: &[bool], ny: usize, nx: usize, y: isize, x: isize) -> u8 {
    if y < 0 || y >= ny as isize || x < 0 || x >= nx as isize {
        0
    } else {
        mask[y as usize * nx + x as usize] as u8
    }
}

/// Count 0â†’1 transitions in the cyclic neighbor sequence Pâ‚‚..Pâ‚‰,Pâ‚‚.
#[inline]
fn transitions(nb: &[u8; 8]) -> u8 {
    let mut count = 0u8;
    for i in 0..8 {
        if nb[i] == 0 && nb[(i + 1) % 8] == 1 {
            count += 1;
        }
    }
    count
}

/// Sub-iteration selector for Zhangâ€“Suen thinning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZhangSuenPass {
    /// Sub-iteration 1: remove from south-east border pixels.
    Pass1,
    /// Sub-iteration 2: remove from north-west border pixels.
    Pass2,
}

/// One sub-iteration of Zhangâ€“Suen. Returns the number of pixels removed.
fn zhang_suen_step(mask: &mut [bool], ny: usize, nx: usize, pass: ZhangSuenPass) -> usize {
    // Heuristic: thinning typically removes a fraction of border voxels per step.
    let mut to_remove: Vec<usize> = Vec::with_capacity(ny * nx / 16);
    for iy in 0..ny {
        for ix in 0..nx {
            if !mask[iy * nx + ix] {
                continue;
            }
            let y = iy as isize;
            let x = ix as isize;
            // Clockwise from north: P2, P3, P4, P5, P6, P7, P8, P9.
            let nb: [u8; 8] = [
                pixel(mask, ny, nx, y - 1, x),     // P2 north
                pixel(mask, ny, nx, y - 1, x + 1), // P3 northeast
                pixel(mask, ny, nx, y, x + 1),     // P4 east
                pixel(mask, ny, nx, y + 1, x + 1), // P5 southeast
                pixel(mask, ny, nx, y + 1, x),     // P6 south
                pixel(mask, ny, nx, y + 1, x - 1), // P7 southwest
                pixel(mask, ny, nx, y, x - 1),     // P8 west
                pixel(mask, ny, nx, y - 1, x - 1), // P9 northwest
            ];
            let b: u8 = nb.iter().sum();
            if !(2..=6).contains(&b) {
                continue;
            }
            if transitions(&nb) != 1 {
                continue;
            }
            let (p2, p4, p6, p8) = (nb[0], nb[2], nb[4], nb[6]);
            if pass == ZhangSuenPass::Pass1 {
                // Sub-iteration 1: P2Â·P4Â·P6 = 0 AND P4Â·P6Â·P8 = 0
                if p2 * p4 * p6 != 0 {
                    continue;
                }
                if p4 * p6 * p8 != 0 {
                    continue;
                }
            } else {
                // Sub-iteration 2: P2Â·P4Â·P8 = 0 AND P2Â·P6Â·P8 = 0
                if p2 * p4 * p8 != 0 {
                    continue;
                }
                if p2 * p6 * p8 != 0 {
                    continue;
                }
            }
            to_remove.push(iy * nx + ix);
        }
    }
    let count = to_remove.len();
    for idx in to_remove {
        mask[idx] = false;
    }
    count
}

/// Zhangâ€“Suen iterative thinning for 2-D binary images.
pub(super) fn zhang_suen(flat: &[f32], ny: usize, nx: usize) -> Vec<f32> {
    let mut mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();
    loop {
        let removed1 = zhang_suen_step(&mut mask, ny, nx, ZhangSuenPass::Pass1);
        let removed2 = zhang_suen_step(&mut mask, ny, nx, ZhangSuenPass::Pass2);
        if removed1 == 0 && removed2 == 0 {
            break;
        }
    }
    mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
}
