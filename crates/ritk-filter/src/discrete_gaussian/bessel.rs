// Abramowitz & Stegun 9.8.1–9.8.4 polynomial approximations (the exact forms
// ITK uses), so the discrete Gaussian kernel is float-exact to SimpleITK.

/// Modified Bessel function of the first kind, order 0.
pub(super) fn modified_bessel_i0(y: f64) -> f64 {
    let d = y.abs();
    if d < 3.75 {
        let m = (y / 3.75) * (y / 3.75);
        1.0 + m
            * (3.5156229
                + m * (3.0899424
                    + m * (1.2067492 + m * (0.2659732 + m * (0.0360768 + m * 0.0045813)))))
    } else {
        let m = 3.75 / d;
        (d.exp() / d.sqrt())
            * (0.39894228
                + m * (0.01328592
                    + m * (0.00225319
                        + m * (-0.00157565
                            + m * (0.00916281
                                + m * (-0.02057706
                                    + m * (0.02635537 + m * (-0.01647633 + m * 0.00392377))))))))
    }
}

/// Modified Bessel function of the first kind, order 1.
pub(super) fn modified_bessel_i1(y: f64) -> f64 {
    let d = y.abs();
    let acc = if d < 3.75 {
        let m = (y / 3.75) * (y / 3.75);
        d * (0.5
            + m * (0.87890594
                + m * (0.51498869
                    + m * (0.15084934 + m * (0.02658733 + m * (0.00301532 + m * 0.00032411))))))
    } else {
        let m = 3.75 / d;
        let a = 0.02282967 + m * (-0.02895312 + m * (0.01787654 - m * 0.00420059));
        let a = 0.39894228
            + m * (-0.03988024 + m * (-0.00362018 + m * (0.00163801 + m * (-0.01031555 + m * a))));
        a * (d.exp() / d.sqrt())
    };
    if y < 0.0 {
        -acc
    } else {
        acc
    }
}

/// Modified Bessel function of the first kind, order `n ≥ 2`, via Miller's
/// downward recurrence seeded from `j = 2·(n + √(40n))` and renormalised by
/// `I0` (Numerical Recipes / ITK `ModifiedBesselI`).
pub(super) fn modified_bessel_i(n: usize, y: f64) -> f64 {
    if n == 0 {
        return modified_bessel_i0(y);
    }
    if n == 1 {
        return modified_bessel_i1(y);
    }
    if y == 0.0 {
        return 0.0;
    }
    let tox = 2.0 / y.abs();
    let (mut bip, mut bi, mut ans) = (0.0_f64, 1.0_f64, 0.0_f64);
    let mut j = 2 * (n + (40.0 * n as f64).sqrt() as usize);
    while j > 0 {
        let bim = bip + j as f64 * tox * bi;
        bip = bi;
        bi = bim;
        if bi.abs() > 1.0e10 {
            bi *= 1.0e-10;
            bip *= 1.0e-10;
            ans *= 1.0e-10;
        }
        if j == n {
            ans = bip;
        }
        j -= 1;
    }
    ans *= modified_bessel_i0(y) / bi;
    if y < 0.0 && n % 2 == 1 {
        -ans
    } else {
        ans
    }
}
