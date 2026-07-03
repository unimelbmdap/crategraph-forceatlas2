//! JS-compatible transcendentals. V8's Math.log/Math.pow are fdlibm-derived
//! (v8/src/base/ieee754.cc); the `libm` crate shares that lineage for `log`,
//! confirmed bit-for-bit against V8's `Math.log` golden values (see the test
//! below). `libm::pow`, however, was found to diverge from V8's `Math.pow`
//! by 1 ULP on `pow(7, 1.5)`: V8 still ships fdlibm's classic
//! double-precision `pow` algorithm, while the `libm` crate's `pow` derives
//! from a different (musl-descended) implementation. To keep bit-exact
//! parity with V8, `pow` below is a direct Rust port of V8's
//! `v8::base::ieee754::legacy::pow` (`src/base/ieee754.cc`, V8 tag
//! 13.6.233.10 -- the version bundled with the Node v24.3.0 used to
//! generate `tests/node_reference/fixtures/math_goldens.json`), retaining
//! the original Sun/fdlibm licence notice below.
//!
//! Kernel code must call `js_math::log` / `js_math::pow` for these two
//! operations, never `f64::ln` / `f64::powf` (platform libm varies by OS).
//! `f64::sqrt` remains fine to call directly: it is a single correctly
//! rounded hardware instruction on every target we support, not a
//! multi-step transcendental approximation.

pub use libm::log;

// The `pow` implementation below is adapted from fdlibm
// (http://www.netlib.org/fdlibm), by way of V8's
// `src/base/ieee754.cc` (`v8::base::ieee754::legacy::pow`).
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// The original source code covered by the above licence has been modified
// significantly by Google Inc. for V8, and ported from C++ to Rust here.

/// Split a `f64` into its high/low 32-bit words, matching V8's
/// `EXTRACT_WORDS` macro.
#[inline]
fn extract_words(d: f64) -> (i32, u32) {
    let bits = d.to_bits();
    ((bits >> 32) as u32 as i32, (bits & 0xFFFF_FFFF) as u32)
}

/// Read the high 32-bit word of a `f64`, matching `GET_HIGH_WORD`.
#[inline]
fn get_high_word(d: f64) -> i32 {
    (d.to_bits() >> 32) as u32 as i32
}

/// Replace the high 32-bit word of a `f64`, matching `SET_HIGH_WORD`.
#[inline]
fn set_high_word(d: f64, v: i32) -> f64 {
    let mut bits = d.to_bits();
    bits &= 0x0000_0000_FFFF_FFFF;
    bits |= (v as u32 as u64) << 32;
    f64::from_bits(bits)
}

/// Replace the low 32-bit word of a `f64`, matching `SET_LOW_WORD`.
#[inline]
fn set_low_word(d: f64, v: u32) -> f64 {
    let mut bits = d.to_bits();
    bits &= 0xFFFF_FFFF_0000_0000;
    bits |= v as u64;
    f64::from_bits(bits)
}

/// Return `x` raised to the `y`th power, bit-identical to V8's `Math.pow`.
///
/// ES2019 Draft 2019-01-02 12.6.4, Math.pow & Exponentiation Operator.
///
/// Method:
///     Let x =  2   * (1+f)
///     1. Compute and return log2(x) in two pieces:
///        log2(x) = w1 + w2,
///        where w1 has 53-24 = 29 bit trailing zeros.
///     2. Perform y*log2(x) = n+y' by simulating multi-precision
///        arithmetic, where |y'|<=0.5.
///     3. Return x**y = 2**n*exp(y'*log2)
///
/// Special cases:
///     1.  (anything) ** 0  is 1
///     2.  (anything) ** 1  is itself
///     3.  (anything) ** NAN is NAN
///     4.  NAN ** (anything except 0) is NAN
///     5.  +-(|x| > 1) **  +INF is +INF
///     6.  +-(|x| > 1) **  -INF is +0
///     7.  +-(|x| < 1) **  +INF is +0
///     8.  +-(|x| < 1) **  -INF is +INF
///     9.  +-1         ** +-INF is NAN
///     10. +0 ** (+anything except 0, NAN)               is +0
///     11. -0 ** (+anything except 0, NAN, odd integer)  is +0
///     12. +0 ** (-anything except 0, NAN)               is +INF
///     13. -0 ** (-anything except 0, NAN, odd integer)  is +INF
///     14. -0 ** (odd integer) = -( +0 ** (odd integer) )
///     15. +INF ** (+anything except 0,NAN) is +INF
///     16. +INF ** (-anything except 0,NAN) is +0
///     17. -INF ** (anything)  = -0 ** (-anything)
///     18. (-anything) ** (integer) is (-1)**(integer)*(+anything**integer)
///     19. (-anything except 0 and inf) ** (non-integer) is NAN
///
/// Accuracy:
///      pow(x,y) returns x**y nearly rounded. In particular,
///      pow(integer, integer) always returns the correct integer provided
///      it is representable.
#[allow(clippy::all)]
pub fn pow(x: f64, y: f64) -> f64 {
    const BP: [f64; 2] = [1.0, 1.5];
    const DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01]; // 0x3FE2B803, 0x40000000
    const DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08]; // 0x3E4CFDEB, 0x43CFD006
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const TWO53: f64 = 9007199254740992.0; // 0x43400000, 0x00000000
    const HUGE: f64 = 1.0e300;
    const TINY: f64 = 1.0e-300;
    // poly coefs for (3/2)*(log(x)-2s-2/3*s**3
    const L1: f64 = 5.99999999999994648725e-01; // 0x3FE33333, 0x33333303
    const L2: f64 = 4.28571428578550184252e-01; // 0x3FDB6DB6, 0xDB6FABFF
    const L3: f64 = 3.33333329818377432918e-01; // 0x3FD55555, 0x518F264D
    const L4: f64 = 2.72728123808534006489e-01; // 0x3FD17460, 0xA91D4101
    const L5: f64 = 2.30660745775561754067e-01; // 0x3FCD864A, 0x93C9DB65
    const L6: f64 = 2.06975017800338417784e-01; // 0x3FCA7E28, 0x4A454EEF
    const P1: f64 = 1.66666666666666019037e-01; // 0x3FC55555, 0x5555553E
    const P2: f64 = -2.77777777770155933842e-03; // 0xBF66C16C, 0x16BEBD93
    const P3: f64 = 6.61375632143793436117e-05; // 0x3F11566A, 0xAF25DE2C
    const P4: f64 = -1.65339022054652515390e-06; // 0xBEBBBD41, 0xC5D26BF1
    const P5: f64 = 4.13813679705723846039e-08; // 0x3E663769, 0x72BEA4D0
    const LG2: f64 = 6.93147180559945286227e-01; // 0x3FE62E42, 0xFEFA39EF
    const LG2_H: f64 = 6.93147182464599609375e-01; // 0x3FE62E43, 0x00000000
    const LG2_L: f64 = -1.90465429995776804525e-09; // 0xBE205C61, 0x0CA86C39
    const OVT: f64 = 8.0085662595372944372e-17; // -(1024-log2(ovfl+.5ulp))
    const CP: f64 = 9.61796693925975554329e-01; // 0x3FEEC709, 0xDC3A03FD =2/(3ln2)
    const CP_H: f64 = 9.61796700954437255859e-01; // 0x3FEEC709, 0xE0000000 =(float)cp
    const CP_L: f64 = -7.02846165095275826516e-09; // 0xBE3E2FE0, 0x145B01F5 =tail cp_h
    const IVLN2: f64 = 1.44269504088896338700e+00; // 0x3FF71547, 0x652B82FE =1/ln2
    const IVLN2_H: f64 = 1.44269502162933349609e+00; // 0x3FF71547, 0x60000000 =24b 1/ln2
    const IVLN2_L: f64 = 1.92596299112661746887e-08; // 0x3E54AE0B, 0xF85DDF44 =1/ln2 tail

    let (hx, lx) = extract_words(x);
    let (hy, ly) = extract_words(y);
    let mut ix = hx & 0x7fff_ffff;
    let iy = hy & 0x7fff_ffff;

    // y==zero: x**0 = 1
    if iy == 0 && ly == 0 {
        return ONE;
    }

    // +-NaN return x+y
    if ix > 0x7ff0_0000
        || (ix == 0x7ff0_0000 && lx != 0)
        || iy > 0x7ff0_0000
        || (iy == 0x7ff0_0000 && ly != 0)
    {
        return x + y;
    }

    // determine if y is an odd int when x < 0
    // yisint = 0 ... y is not an integer
    // yisint = 1 ... y is an odd int
    // yisint = 2 ... y is an even int
    let mut yisint = 0i32;
    if hx < 0 {
        if iy >= 0x4340_0000 {
            yisint = 2; // even integer y
        } else if iy >= 0x3ff0_0000 {
            let k = (iy >> 20) - 0x3ff; // exponent
            if k > 20 {
                let shift = (52 - k) as u32;
                let j = (ly >> shift) as i32;
                if (j.wrapping_shl(shift)) == ly as i32 {
                    yisint = 2 - (j & 1);
                }
            } else if ly == 0 {
                let shift = (20 - k) as u32;
                let j = iy >> shift;
                if (j.wrapping_shl(shift)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }

    // special value of y
    if ly == 0 {
        if iy == 0x7ff0_0000 {
            // y is +-inf
            return if ix == 0x3ff0_0000 && lx == 0 {
                y - y // inf**+-1 is NaN
            } else if ix >= 0x3ff0_0000 {
                // (|x|>1)**+-inf = inf,0
                if hy >= 0 {
                    y
                } else {
                    ZERO
                }
            } else {
                // (|x|<1)**-,+inf = inf,0
                if hy < 0 {
                    -y
                } else {
                    ZERO
                }
            };
        }
        if iy == 0x3ff0_0000 {
            // y is  +-1
            return if hy < 0 { ONE / x } else { x };
        }
        if hy == 0x4000_0000 {
            return x * x; // y is  2
        }
        if hy == 0x3fe0_0000 && hx >= 0 {
            // y is  0.5, x >= +0
            return x.sqrt();
        }
    }

    let mut ax = x.abs();
    // special value of x
    if lx == 0 && (ix == 0x7ff0_0000 || ix == 0 || ix == 0x3ff0_0000) {
        let mut z = ax; // x is +-0,+-inf,+-1
        if hy < 0 {
            z = ONE / z; // z = (1/|x|)
        }
        if hx < 0 {
            if ix == 0x3ff0_0000 && yisint == 0 {
                // (-1)**non-int is NaN. V8 uses a raw signaling-NaN bit
                // pattern here; we return the standard quiet NaN instead.
                // JS Number NaNs are observationally indistinguishable, and
                // this branch (negative base, non-integer exponent) is not
                // exercised by the golden fixtures.
                z = f64::NAN;
            } else if yisint == 1 {
                z = -z; // (x<0)**odd = -(|x|**odd)
            }
        }
        return z;
    }

    let n0 = (hx >> 31) + 1;

    // (x<0)**(non-int) is NaN
    if n0 == 0 && yisint == 0 {
        return f64::NAN;
    }

    let mut s = ONE; // s (sign of result -ve**odd) = -1 else = 1
    if n0 == 0 && yisint == 1 {
        s = -ONE; // (-ve)**(odd int)
    }

    // |y| is huge
    let t1: f64;
    let t2: f64;
    if iy > 0x41e0_0000 {
        // if |y| > 2**31
        if iy > 0x43f0_0000 {
            // if |y| > 2**64, must o/uflow
            if ix <= 0x3fef_ffff {
                return if hy < 0 { HUGE * HUGE } else { TINY * TINY };
            }
            if ix >= 0x3ff0_0000 {
                return if hy > 0 { HUGE * HUGE } else { TINY * TINY };
            }
        }
        // over/underflow if x is not close to one
        if ix < 0x3fef_ffff {
            return if hy < 0 { s * HUGE * HUGE } else { s * TINY * TINY };
        }
        if ix > 0x3ff0_0000 {
            return if hy > 0 { s * HUGE * HUGE } else { s * TINY * TINY };
        }
        // now |1-x| is tiny <= 2**-20, suffice to compute
        // log(x) by x-x^2/2+x^3/3-x^4/4
        let t = ax - ONE; // t has 20 trailing zeros
        let w = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25));
        let u = IVLN2_H * t; // ivln2_h has 21 sig. bits
        let v = t * IVLN2_L - w * IVLN2;
        let mut t1_ = u + v;
        t1_ = set_low_word(t1_, 0);
        let t2_ = v - (t1_ - u);
        t1 = t1_;
        t2 = t2_;
    } else {
        let mut n = 0i32;
        // take care subnormal number
        if ix < 0x0010_0000 {
            ax *= TWO53;
            n -= 53;
            ix = get_high_word(ax);
        }
        n = n.wrapping_add((ix >> 20) - 0x3ff);
        let j = ix & 0x000f_ffff;
        // determine interval
        ix = j | 0x3ff0_0000; // normalize ix
        let k: i32;
        if j <= 0x0003_988e {
            k = 0; // |x|<sqrt(3/2)
        } else if j < 0x000b_b67a {
            k = 1; // |x|<sqrt(3)
        } else {
            k = 0;
            n = n.wrapping_add(1);
            ix -= 0x0010_0000;
        }
        ax = set_high_word(ax, ix);

        // compute ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5)
        let u = ax - BP[k as usize]; // bp[0]=1.0, bp[1]=1.5
        let v = ONE / (ax + BP[k as usize]);
        let ss = u * v;
        let mut s_h = ss;
        s_h = set_low_word(s_h, 0);
        // t_h=ax+bp[k] High
        let mut t_h = ZERO;
        t_h = set_high_word(
            t_h,
            ((ix >> 1) | 0x2000_0000)
                .wrapping_add(0x0008_0000)
                .wrapping_add(k << 18),
        );
        let t_l = ax - (t_h - BP[k as usize]);
        let s_l = v * ((u - s_h * t_h) - s_h * t_l);
        // compute log(ax)
        let mut s2 = ss * ss;
        let mut r =
            s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        s2 = s_h * s_h;
        t_h = 3.0 + s2 + r;
        t_h = set_low_word(t_h, 0);
        let t_l = r - ((t_h - 3.0) - s2);
        // u+v = ss*(1+...)
        let u = s_h * t_h;
        let v = s_l * t_h + t_l * ss;
        // 2/(3log2)*(ss+...)
        let mut p_h = u + v;
        p_h = set_low_word(p_h, 0);
        let p_l = v - (p_h - u);
        let z_h = CP_H * p_h; // cp_h+cp_l = 2/(3*log2)
        let z_l = CP_L * p_h + p_l * CP + DP_L[k as usize];
        // log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l
        let t = n as f64;
        let mut t1_ = ((z_h + z_l) + DP_H[k as usize]) + t;
        t1_ = set_low_word(t1_, 0);
        let t2_ = z_l - (((t1_ - t) - DP_H[k as usize]) - z_h);
        t1 = t1_;
        t2 = t2_;
    }

    // split up y into y1+y2 and compute (y1+y2)*(t1+t2)
    let mut y1 = y;
    y1 = set_low_word(y1, 0);
    let p_l = (y - y1) * t1 + y * t2;
    let p_h = y1 * t1;
    let z = p_l + p_h;
    let (mut j, i_lo) = extract_words(z);
    if j >= 0x4090_0000 {
        // z >= 1024
        if j != 0x4090_0000 || i_lo != 0 {
            // z > 1024
            return s * HUGE * HUGE; // overflow
        }
        if p_l + OVT > z - p_h {
            return s * HUGE * HUGE; // overflow
        }
    } else if (j & 0x7fff_ffff) >= 0x4090_cc00 {
        // z <= -1075
        if j != 0xc090_cc00u32 as i32 || i_lo != 0 {
            // z < -1075
            return s * TINY * TINY; // underflow
        }
        if p_l <= z - p_h {
            return s * TINY * TINY; // underflow
        }
    }

    // compute 2**(p_h+p_l)
    let abs_j = j & 0x7fff_ffff;
    let mut k = (abs_j >> 20) - 0x3ff;
    let mut n = 0i32;
    let mut p_h = p_h;
    if abs_j > 0x3fe0_0000 {
        // if |z| > 0.5, set n = [z+0.5]
        n = j.wrapping_add(0x0010_0000 >> (k + 1));
        k = ((n & 0x7fff_ffff) >> 20) - 0x3ff; // new k for n
        let mut t = ZERO;
        t = set_high_word(t, n & !(0x000f_ffff >> k));
        n = ((n & 0x000f_ffff) | 0x0010_0000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let mut t = p_l + p_h;
    t = set_low_word(t, 0);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    let z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let r = (z * t1) / ((t1 - TWO) - (w + z * w));
    let z = ONE - (r - z);
    j = get_high_word(z);
    let delta = ((n as u32) << 20) as i32;
    j = j.wrapping_add(delta);
    let z = if (j >> 20) <= 0 {
        libm::scalbn(z, n) // subnormal output
    } else {
        let tmp = get_high_word(z);
        set_high_word(z, tmp.wrapping_add(delta))
    };
    s * z
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_v8_bit_for_bit() {
        let raw = include_str!("../../tests/node_reference/fixtures/math_goldens.json");
        // minimal hand-rolled parse to avoid a serde dep: the file is small and
        // regular; extract every quoted decimal u64 in order.
        let nums: Vec<u64> = raw
            .split('"')
            .filter_map(|s| s.parse::<u64>().ok())
            .collect();
        // layout: log pairs come first (2 numbers each, 9 pairs), then pow triples.
        let (log_part, pow_part) = nums.split_at(9 * 2);
        let mut failures = Vec::new();
        for c in log_part.chunks(2) {
            let (x, want) = (f64::from_bits(c[0]), c[1]);
            let got = super::log(x).to_bits();
            if got != want {
                failures.push(format!("log({x}) got={got} want={want}"));
            }
        }
        for c in pow_part.chunks(3) {
            let (x, y, want) = (f64::from_bits(c[0]), f64::from_bits(c[1]), c[2]);
            let got = super::pow(x, y).to_bits();
            if got != want {
                failures.push(format!("pow({x},{y}) got={got} want={want}"));
            }
        }
        assert!(failures.is_empty(), "bit mismatches: {failures:#?}");
    }
}
