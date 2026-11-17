// Browsers round Math.exp, tanh, log, sin and cos differently in the last digit, and training magnifies that.
// These are the fdlibm routines in + - * / only, which every browser rounds the same, so seeded demos agree.
(function pinMath() {
  const f64 = new Float64Array(1), u32 = new Uint32Array(f64.buffer);
  const HI = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1 ? 1 : 0, LO = 1 - HI;
  const native = { sin: Math.sin, cos: Math.cos };
  const LN2_HI = 6.93147180369123816490e-01, LN2_LO = 1.90821492927058770002e-10, INV_LN2 = 1.44269504088896338700e+00;
  const EXP_MAX = 7.09782712893383973096e+02, EXP_MIN = -7.45133219101941108420e+02;
  const TWO_1023 = 8.98846567431157953865e+307, TWO_M1000 = 9.33263618503218878990e-302, TWO_54 = 1.80143985094819840000e+16;
  const P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03, P3 = 6.61375632143793436117e-05;
  const P4 = -1.65339022054652515390e-06, P5 = 4.13813679705723846039e-08;
  const Q1 = -3.33333333333331316428e-02, Q2 = 1.58730158725481460165e-03, Q3 = -7.93650757867487942473e-05;
  const Q4 = 4.00821782732936239552e-06, Q5 = -2.01099218183624371326e-07;
  const LG1 = 6.666666666666735130e-01, LG2 = 3.999999999940941908e-01, LG3 = 2.857142874366239149e-01, LG4 = 2.222219843214978396e-01;
  const LG5 = 1.818357216161805012e-01, LG6 = 1.531383769920937332e-01, LG7 = 1.479819860511658591e-01;
  const S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03, S3 = -1.98412698298579493134e-04;
  const S4 = 2.75573137070700676789e-06, S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
  const C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03, C3 = 2.48015872894767294178e-05;
  const C4 = -2.75573143513906633035e-07, C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
  const INV_PIO2 = 6.36619772367581382433e-01, PIO2_1 = 1.57079632673412561417e+00, PIO2_1T = 6.07710050650619224932e-11;
  const PIO2_2 = 6.07710050630396597660e-11, PIO2_2T = 2.02226624879595063154e-21;
  const PIO2_3 = 2.02226624871116645580e-21, PIO2_3T = 8.47842766036889956997e-32;
  let tail = 0, turn = 0;

  function high(x) { f64[0] = x; return u32[HI] | 0; }
  function scale(y, k) { f64[0] = y; u32[HI] += k << 20; return f64[0]; }

  function exp(x) {
    let hx = high(x), k = 0, hi = 0, lo = 0;
    const negative = hx < 0 ? 1 : 0;
    hx &= 0x7fffffff;
    if (hx >= 0x40862E42) {
      if (hx >= 0x7ff00000) return x !== x ? x : negative ? 0 : x;
      if (x > EXP_MAX) return Infinity;
      if (x < EXP_MIN) return 0;
    }
    if (hx > 0x3fd62e42) {
      if (hx < 0x3FF0A2B2) {
        hi = negative ? x + LN2_HI : x - LN2_HI;
        lo = negative ? -LN2_LO : LN2_LO;
        k = 1 - negative - negative;
      } else {
        k = (INV_LN2 * x + (negative ? -0.5 : 0.5)) | 0;
        hi = x - k * LN2_HI;
        lo = k * LN2_LO;
      }
      x = hi - lo;
    } else if (hx < 0x3e300000) {
      return 1 + x;
    }
    const t = x * x, c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if (k === 0) return 1 - ((x * c) / (c - 2) - x);
    const y = 1 - ((lo - (x * c) / (2 - c)) - hi);
    if (k >= -1021) return k === 1024 ? y * 2 * TWO_1023 : scale(y, k);
    return scale(y, k + 1000) * TWO_M1000;
  }

  function expm1(x) {
    let hx = high(x), k = 0, hi = 0, lo = 0, c = 0;
    const negative = hx < 0;
    hx &= 0x7fffffff;
    if (hx >= 0x4043687A) {
      if (hx >= 0x40862E42) {
        if (hx >= 0x7ff00000) return x !== x ? x : negative ? -1 : x;
        if (x > EXP_MAX) return Infinity;
      }
      if (negative) return -1;
    }
    if (hx > 0x3fd62e42) {
      if (hx < 0x3FF0A2B2) {
        hi = negative ? x + LN2_HI : x - LN2_HI;
        lo = negative ? -LN2_LO : LN2_LO;
        k = negative ? -1 : 1;
      } else {
        k = (INV_LN2 * x + (negative ? -0.5 : 0.5)) | 0;
        hi = x - k * LN2_HI;
        lo = k * LN2_LO;
      }
      x = hi - lo;
      c = (hi - x) - lo;
    } else if (hx < 0x3c900000) {
      return x;
    }
    const hfx = 0.5 * x, hxs = x * hfx;
    const r1 = 1 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3 - r1 * hfx, e = hxs * ((r1 - t) / (6 - x * t)), y;
    if (k === 0) return x - (x * e - hxs);
    e = x * (e - c) - c;
    e -= hxs;
    if (k === -1) return 0.5 * (x - e) - 0.5;
    if (k === 1) return x < -0.25 ? -2 * (e - (x + 0.5)) : 1 + 2 * (x - e);
    if (k <= -2 || k > 56) {
      y = 1 - (e - x);
      return (k === 1024 ? y * 2 * TWO_1023 : scale(y, k)) - 1;
    }
    if (k < 20) {
      f64[0] = 1;
      u32[HI] = 0x3ff00000 - (0x200000 >> k);
      return scale(f64[0] - (e - x), k);
    }
    f64[0] = 1;
    u32[HI] = (0x3ff - k) << 20;
    t = f64[0];
    y = x - (e + t);
    return scale(y + 1, k);
  }

  function tanh(x) {
    const jx = high(x), ix = jx & 0x7fffffff;
    let z;
    if (ix >= 0x7ff00000) return jx >= 0 ? 1 / x + 1 : 1 / x - 1;
    if (ix < 0x40360000) {
      if (ix < 0x3c800000) return x * (1 + x);
      const a = x < 0 ? -x : x;
      if (ix >= 0x3ff00000) {
        z = 1 - 2 / (expm1(2 * a) + 2);
      } else {
        const t = expm1(-2 * a);
        z = -t / (t + 2);
      }
    } else {
      z = 1;
    }
    return jx >= 0 ? z : -z;
  }

  function log(x) {
    f64[0] = x;
    let hx = u32[HI] | 0, k = 0;
    if (hx < 0x00100000) {
      if (((hx & 0x7fffffff) | u32[LO]) === 0) return -Infinity;
      if (hx < 0) return NaN;
      k -= 54;
      f64[0] = x * TWO_54;
      hx = u32[HI] | 0;
    }
    if (hx >= 0x7ff00000) return x + x;
    k += (hx >> 20) - 1023;
    hx &= 0x000fffff;
    let i = (hx + 0x95f64) & 0x100000;
    u32[HI] = hx | (i ^ 0x3ff00000);
    k += i >> 20;
    const f = f64[0] - 1;
    if ((0x000fffff & (2 + hx)) < 3) {
      if (f === 0) return k === 0 ? 0 : k * LN2_HI + k * LN2_LO;
      const r = f * f * (0.5 - 0.33333333333333333 * f);
      return k === 0 ? f - r : k * LN2_HI - ((r - k * LN2_LO) - f);
    }
    const s = f / (2 + f), z = s * s, w = z * z;
    const t1 = w * (LG2 + w * (LG4 + w * LG6)), t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7))), r = t2 + t1;
    i = (hx - 0x6147a) | (0x6b851 - hx);
    if (i > 0) {
      const hfsq = 0.5 * f * f;
      return k === 0 ? f - (hfsq - s * (hfsq + r)) : k * LN2_HI - ((hfsq - (s * (hfsq + r) + k * LN2_LO)) - f);
    }
    return k === 0 ? f - s * (f - r) : k * LN2_HI - ((s * (f - r) - k * LN2_LO) - f);
  }

  function sinKernel(x, y, hasTail) {
    if ((high(x) & 0x7fffffff) < 0x3e400000) return x;
    const z = x * x, v = z * x, r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    return hasTail ? x - ((z * (0.5 * y - v * r) - y) - v * S1) : x + v * (S1 + z * r);
  }

  function cosKernel(x, y) {
    const ix = high(x) & 0x7fffffff;
    if (ix < 0x3e400000) return 1;
    const z = x * x, r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if (ix < 0x3FD33333) return 1 - (0.5 * z - (z * r - x * y));
    let qx = 0.28125;
    if (ix <= 0x3fe90000) {
      u32[HI] = ix - 0x00200000;
      u32[LO] = 0;
      qx = f64[0];
    }
    return (1 - qx) - ((0.5 * z - qx) - (z * r - x * y));
  }

  // Returns x minus the nearest multiple of pi/2, leaving the rounding error in `tail` and the quarter-turn in `turn`.
  function reduce(x) {
    const hx = high(x), ix = hx & 0x7fffffff, a = x < 0 ? -x : x;
    const n = (a * INV_PIO2 + 0.5) | 0, j = ix >> 20;
    let r = a - n * PIO2_1, w = n * PIO2_1T, y = r - w;
    if (j - ((high(y) >> 20) & 0x7ff) > 16) {
      let t = r;
      w = n * PIO2_2;
      r = t - w;
      w = n * PIO2_2T - ((t - r) - w);
      y = r - w;
      if (j - ((high(y) >> 20) & 0x7ff) > 49) {
        t = r;
        w = n * PIO2_3;
        r = t - w;
        w = n * PIO2_3T - ((t - r) - w);
        y = r - w;
      }
    }
    tail = (r - y) - w;
    turn = n & 3;
    if (hx < 0) {
      y = -y;
      tail = -tail;
      turn = -n & 3;
    }
    return y;
  }

  function sin(x) {
    const ix = high(x) & 0x7fffffff;
    if (ix <= 0x3fe921fb) return sinKernel(x, 0, false);
    if (ix > 0x413921fb) return native.sin(x);
    const y = reduce(x);
    return turn === 0 ? sinKernel(y, tail, true) : turn === 1 ? cosKernel(y, tail) : turn === 2 ? -sinKernel(y, tail, true) : -cosKernel(y, tail);
  }

  function cos(x) {
    const ix = high(x) & 0x7fffffff;
    if (ix <= 0x3fe921fb) return cosKernel(x, 0);
    if (ix > 0x413921fb) return native.cos(x);
    const y = reduce(x);
    return turn === 0 ? cosKernel(y, tail) : turn === 1 ? -sinKernel(y, tail, true) : turn === 2 ? -cosKernel(y, tail) : sinKernel(y, tail, true);
  }

  Math.exp = exp;
  Math.expm1 = expm1;
  Math.tanh = tanh;
  Math.log = log;
  Math.sin = sin;
  Math.cos = cos;
})();
