def _quad_simpsons_mem(f, a, fa, b, fb):
    """Evaluates the Simpson's Rule, also returning m and f(m) to reuse"""
    m = (a + b) / 2
    fm = f(m)
    return (m, fm, abs(b - a) / 6 * (fa + 4 * fm + fb))

def _quad_asr(f, a, fa, b, fb, eps, whole, m, fm):
    """
    Efficient recursive implementation of adaptive Simpson's rule.
    Function values at the start, middle, end of the intervals are retained.
    """
    lm, flm, left  = _quad_simpsons_mem(f, a, fa, m, fm)
    rm, frm, right = _quad_simpsons_mem(f, m, fm, b, fb)
    delta = left + right - whole
    if abs(delta) <= 15 * eps:
        return left + right + delta / 15
    return _quad_asr(f, a, fa, m, fm, eps/2, left , lm, flm) +\
           _quad_asr(f, m, fm, b, fb, eps/2, right, rm, frm)

def quad_asr(f, a, b, eps):
    """Integrate f from a to b using Adaptive Simpson's Rule with max error of eps."""
    fa, fb = f(a), f(b)
    m, fm, whole = _quad_simpsons_mem(f, a, fa, b, fb)
    return _quad_asr(f, a, fa, b, fb, eps, whole, m, fm)

import numpy as np
def f(x):
    return (x**2 - 1)/((x**2 + 1)*(np.sqrt(x**4 + 1)))
print(quad_asr(f, -4*np.pi, 4*np.pi, 1e-12))