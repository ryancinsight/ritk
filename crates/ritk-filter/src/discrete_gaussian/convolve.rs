/// Convolve a 1-D slice with replicate (edge) boundary padding.
#[inline]
fn conv1d_replicate(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }
    let ksz = kernel.len();
    let r = (ksz / 2) as isize;
    let n_i = n as isize;
    output.fill(0.0);
    for (kj, &w) in kernel.iter().enumerate() {
        let offset = kj as isize - r;
        let i_start = ((-offset).max(0) as usize).min(n);
        let i_end = ((n_i - offset).max(0).min(n_i) as usize).min(n);
        if i_start > 0 {
            let left_val = input[0] * w;
            for o in &mut output[..i_start] {
                *o += left_val;
            }
        }
        for i in i_start..i_end {
            output[i] += input[(i as isize + offset) as usize] * w;
        }
        if i_end < n {
            let right_val = input[n - 1] * w;
            for o in &mut output[i_end..] {
                *o += right_val;
            }
        }
    }
}

/// Convolve flat C-order `[NZ, NY, NX]` array along one axis. Rayon-parallel.
fn convolve3d_dim(
    src: &[f32],
    dst: &mut [f32],
    nz: usize,
    ny: usize,
    nx: usize,
    dim: usize,
    kernel: &[f32],
) {
    let nyx = ny * nx;
    match dim {
        2 => {
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                dst,
                nx,
                |ci, o| {
                    let i = &src[ci * nx..ci * nx + o.len()];
                    conv1d_replicate(i, kernel, o);
                },
            );
        }
        1 => {
            let ksz = kernel.len();
            let r = (ksz / 2) as isize;
            let ny_i = ny as isize;
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                dst,
                nyx,
                |ci, os| {
                    let is = &src[ci * nyx..ci * nyx + os.len()];
                    os.fill(0.0);
                    for (kj, &w) in kernel.iter().enumerate() {
                        let r_offset = kj as isize - r;
                        for iy in 0..ny {
                            let sy = ((iy as isize + r_offset).clamp(0, ny_i - 1)) as usize;
                            let src_row = &is[sy * nx..(sy + 1) * nx];
                            let dst_row = &mut os[iy * nx..(iy + 1) * nx];
                            for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
                                *d += s * w;
                            }
                        }
                    }
                },
            );
        }
        0 => {
            let ksz = kernel.len();
            let r = (ksz / 2) as isize;
            let nz_i = nz as isize;
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                dst,
                nyx,
                |iz, out_slice| {
                    out_slice.fill(0.0);
                    for (kj, &w) in kernel.iter().enumerate() {
                        let sz = ((iz as isize + kj as isize - r).clamp(0, nz_i - 1)) as usize;
                        let src_z = &src[sz * nyx..(sz + 1) * nyx];
                        for (o, &s) in out_slice.iter_mut().zip(src_z.iter()) {
                            *o += s * w;
                        }
                    }
                },
            );
        }
        _ => unreachable!(),
    }
}

fn convolve_nd_dim_serial(
    src: &[f32],
    dst: &mut [f32],
    shape: &[usize],
    dim: usize,
    kernel: &[f32],
) {
    let d = shape.len();
    let mut strides = vec![1usize; d];
    for i in (0..d.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let line_len = shape[dim];
    let line_stride = strides[dim];
    let n_total: usize = shape.iter().product();
    let n_lines = n_total / line_len.max(1);
    let ksz = kernel.len();
    let r = (ksz / 2) as isize;
    let ll_i = line_len as isize;
    let mut idx = vec![0usize; d];
    let mut ob = vec![0.0f32; line_len];
    for _line in 0..n_lines {
        let base: usize = idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
        for (i, ob_elem) in ob.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (kj, &w) in kernel.iter().enumerate() {
                let pos = ((i as isize + kj as isize - r).clamp(0, ll_i - 1)) as usize;
                acc += src[base + pos * line_stride] * w;
            }
            *ob_elem = acc;
        }
        for (i, &val) in ob.iter().enumerate() {
            dst[base + i * line_stride] = val;
        }
        let mut carry = true;
        for dd in (0..d).rev() {
            if dd == dim {
                continue;
            }
            if carry {
                idx[dd] += 1;
                if idx[dd] < shape[dd] {
                    carry = false;
                } else {
                    idx[dd] = 0;
                }
            }
        }
    }
}

pub(crate) fn convolve_separable<const D: usize>(
    mut data: Vec<f32>,
    shape: [usize; D],
    kernels: &[Option<Vec<f32>>; D],
) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut buf = vec![0.0f32; n];
    for (dim, kernel_opt) in kernels.iter().enumerate() {
        let Some(kernel) = kernel_opt else {
            continue;
        };
        if D == 3 {
            convolve3d_dim(&data, &mut buf, shape[0], shape[1], shape[2], dim, kernel);
        } else {
            convolve_nd_dim_serial(&data, &mut buf, shape.as_slice(), dim, kernel);
        }
        std::mem::swap(&mut data, &mut buf);
    }
    data
}
