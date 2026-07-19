//! Exact port of `itk::Statistics::NormalVariateGenerator` (Marsaglia–MacLaren
//! *FastNorm*), the deterministic generator behind `AdditiveGaussianNoise`.
//!
//! Reproduces `sitk.AdditiveGaussianNoise` bit-for-bit when seeded identically
//! (`Hash(userSeed, regionStart) = userSeed·2654435761`) and applied in scanline
//! order — verified against SimpleITK's noise sequence.

const SCALE: f64 = 30_000_000.0;
const RSCALE: f64 = 1.0 / SCALE;
const RCONS: f64 = 1.0 / (2.0 * 1024.0 * 1024.0 * 1024.0);
const ELEN: i32 = 7;
const LEN: i32 = 128;
const LMASK: i32 = 4 * (LEN - 1);
const TLEN: usize = 8 * 128;

/// Knuth multiplicative hash combining a user seed with a region offset.
#[inline]
pub(crate) fn hash(a: u32, b: u32) -> u32 {
    a.wrapping_add(b).wrapping_mul(2_654_435_761)
}

#[inline]
fn signed_shift_xor(irs: i32) -> i32 {
    let uirs = irs as u32;
    (if irs <= 0 {
        (uirs << 1) ^ 333_556_017
    } else {
        uirs << 1
    }) as i32
}

/// ITK FastNorm normal-variate generator. Stateful and sequential.
pub(crate) struct FastNorm {
    lseed: i32,
    irs: i32,
    gaussfaze: i32,
    nslew: i32,
    gscale: f64,
    vec1: [i32; TLEN],
    chic1: f64,
    chic2: f64,
    actual_rsd: f64,
}

impl FastNorm {
    /// Initialize from a 32-bit signed seed (ITK `Initialize`).
    pub(crate) fn new(seed: i32) -> Self {
        let fake = 1.0 + 0.125 / TLEN as f64;
        Self {
            lseed: seed,
            irs: seed,
            gaussfaze: 1,
            nslew: 0,
            gscale: RSCALE,
            vec1: [0; TLEN],
            chic2: (2.0 * TLEN as f64 - fake * fake).sqrt() / fake,
            chic1: fake * (0.5 / TLEN as f64).sqrt(),
            actual_rsd: 0.0,
        }
    }

    #[inline]
    fn lcg(&mut self) {
        self.lseed = (69069i64 * self.lseed as i64 + 33331) as i32;
    }

    /// Next normal variate.
    pub(crate) fn variate(&mut self) -> f64 {
        self.gaussfaze -= 1;
        if self.gaussfaze != 0 {
            return self.gscale * self.vec1[self.gaussfaze as usize] as f64;
        }
        self.fastnorm()
    }

    fn fastnorm(&mut self) -> f64 {
        if self.nslew & 0xFF == 0 {
            self.renorm();
        }
        self.nslew = self.nslew.wrapping_add(1);
        self.gaussfaze = TLEN as i32 - 1;
        self.lcg();
        self.irs = signed_shift_xor(self.irs);
        let mut t = (self.irs as i64 + self.lseed as i64) as i32;
        if t < 0 {
            t = !t;
        }
        t >>= 29 - 2 * ELEN;
        let mut skew = (LEN - 1) & t;
        t >>= ELEN;
        skew *= 4;
        let mut stride = (LEN / 2 - 1) & t;
        t >>= ELEN - 1;
        stride = 8 * stride + 4;
        let mtype = (t & 3) as usize;
        let stype = (self.nslew & 3) as usize;

        let len = LEN;
        let (inc, mask, mut pa, mut pb, mut pc, mut pd, p0): (i32, i32, i32, i32, i32, i32, i32) =
            match stype {
                0 => (1, LMASK, 0, len, 2 * len, 3 * len, 4 * len),
                1 => (1, LMASK, 4 * len, 5 * len, 6 * len, 7 * len, 0),
                2 => {
                    skew *= 2;
                    stride *= 2;
                    (2, 2 * LMASK, 1, 1 + 2 * len, 1 + 4 * len, 1 + 6 * len, 0)
                }
                _ => {
                    skew *= 2;
                    stride *= 2;
                    (2, 2 * LMASK, 0, 2 * len, 4 * len, 6 * len, 1)
                }
            };

        match mtype {
            0 => pa += inc * (len - 1),
            1 => pb += inc * (len - 1),
            2 => pc += inc * (len - 1),
            _ => pd += inc * (len - 1),
        }

        let v = &mut self.vec1;
        let mut i = LEN;
        while i != 0 {
            skew = (skew + stride) & mask;
            let mut pe = p0 + skew;
            let (mut p, mut q, mut r, mut s) = match mtype {
                0 => (
                    -v[pa as usize],
                    -v[pb as usize],
                    v[pc as usize],
                    v[pd as usize],
                ),
                1 => (
                    -v[pa as usize],
                    v[pb as usize],
                    v[pc as usize],
                    -v[pd as usize],
                ),
                2 => (
                    v[pa as usize],
                    -v[pb as usize],
                    v[pc as usize],
                    -v[pd as usize],
                ),
                _ => (
                    v[pa as usize],
                    v[pb as usize],
                    -v[pc as usize],
                    -v[pd as usize],
                ),
            };
            let mut t = (p.wrapping_add(q).wrapping_add(r).wrapping_add(s)) >> 1;
            p = t - p;
            q = t - q;
            r = t - r;
            s = t - s;
            t = match mtype {
                1 | 2 => v[pe as usize],
                _ => -v[pe as usize],
            };
            v[pe as usize] = p;
            pe += inc;
            p = match mtype {
                1 => -v[pe as usize],
                _ => v[pe as usize],
            };
            v[pe as usize] = q;
            pe += inc;
            q = match mtype {
                3 => v[pe as usize],
                _ => -v[pe as usize],
            };
            v[pe as usize] = r;
            pe += inc;
            r = match mtype {
                2 | 3 => -v[pe as usize],
                _ => v[pe as usize],
            };
            v[pe as usize] = s;
            s = (p.wrapping_add(q).wrapping_add(r).wrapping_add(t)) >> 1;
            match mtype {
                0 => {
                    v[pa as usize] = s - p;
                    pa -= inc;
                    v[pb as usize] = s - q;
                    pb += inc;
                    v[pc as usize] = s - r;
                    pc += inc;
                    v[pd as usize] = s - t;
                    pd += inc;
                }
                1 => {
                    v[pa as usize] = s - p;
                    pa += inc;
                    v[pb as usize] = s - t;
                    pb -= inc;
                    v[pc as usize] = s - q;
                    pc += inc;
                    v[pd as usize] = s - r;
                    pd += inc;
                }
                2 => {
                    v[pa as usize] = s - r;
                    pa += inc;
                    v[pb as usize] = s - p;
                    pb += inc;
                    v[pc as usize] = s - q;
                    pc -= inc;
                    v[pd as usize] = s - t;
                    pd += inc;
                }
                _ => {
                    v[pa as usize] = s - q;
                    pa += inc;
                    v[pb as usize] = s - r;
                    pb += inc;
                    v[pc as usize] = s - t;
                    pc += inc;
                    v[pd as usize] = s - p;
                    pd -= inc;
                }
            }
            i -= 1;
        }

        let ts = self.chic1 * (self.chic2 + self.gscale * self.vec1[TLEN - 1] as f64);
        self.gscale = RSCALE * ts * self.actual_rsd;
        self.gscale * self.vec1[0] as f64
    }

    fn renorm(&mut self) {
        if self.nslew & 0xFFFF != 0 {
            self.recalc();
            return;
        }
        let mut ts = 0.0f64;
        let mut p = 0usize;
        loop {
            self.lcg();
            self.irs = signed_shift_xor(self.irs);
            let r = (self.irs as i64 + self.lseed as i64) as i32;
            let tx = RCONS * r as f64;
            self.lcg();
            self.irs = signed_shift_xor(self.irs);
            let r = (self.irs as i64 + self.lseed as i64) as i32;
            let ty = RCONS * r as f64;
            let tr = tx * tx + ty * ty;
            // ITK source verbatim: `(tr > 1.0) || (tr < 0.1)` (tr is a finite
            // sum of squares; the range form would obscure the source mapping).
            #[allow(clippy::manual_range_contains)]
            if tr > 1.0 || tr < 0.1 {
                continue;
            }
            self.lcg();
            self.irs = signed_shift_xor(self.irs);
            let mut r = (self.irs as i64 + self.lseed as i64) as i32;
            if r < 0 {
                r = !r;
            }
            let mut tz = -2.0 * ((r as f64 + 0.5) * RCONS).ln();
            ts += tz;
            tz = (tz / tr).sqrt();
            self.vec1[p] = (SCALE * tx * tz) as i32;
            p += 1;
            self.vec1[p] = (SCALE * ty * tz) as i32;
            p += 1;
            if p >= TLEN {
                break;
            }
        }
        ts = TLEN as f64 / ts;
        let tr = ts.sqrt();
        for slot in self.vec1.iter_mut() {
            let tx = *slot as f64 * tr;
            *slot = (if tx < 0.0 { tx - 0.5 } else { tx + 0.5 }) as i32;
        }
        self.recalc();
    }

    fn recalc(&mut self) {
        let mut ts = 0.0f64;
        for &val in self.vec1.iter() {
            let tx = val as f64;
            ts += tx * tx;
        }
        ts = (ts / (SCALE * SCALE * TLEN as f64)).sqrt();
        self.actual_rsd = 1.0 / ts;
    }
}
