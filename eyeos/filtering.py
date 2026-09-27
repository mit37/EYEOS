"""The One Euro Filter (Casiez, Roussel & Vogel, CHI 2012).

This is the canonical reference algorithm: a low-pass filter on the signal whose
cutoff frequency increases with the estimated speed of the signal (itself a low-passed
derivative). At rest it smooths hard (few px of jitter survive); moving fast it lets
the signal through with little lag. ``min_cutoff`` sets the resting cutoff (lower =
smoother, more lag), ``beta`` sets how much speed opens the cutoff up (higher = less
lag during fast motion, at the cost of a bit more jitter while moving).

Reference: http://cristal.univ-lille.fr/~casiez/1euro/
"""

from __future__ import annotations

import math


def smoothing_factor(t_e: float, cutoff: float) -> float:
    """The exponential-smoothing alpha for a cutoff frequency `cutoff` over dt `t_e`."""
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(alpha: float, x: float, x_prev: float) -> float:
    return alpha * x + (1 - alpha) * x_prev


class OneEuroFilter:
    """A 1D One Euro Filter. Call it with successive (timestamp, value) pairs."""

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 0:
            # Non-advancing or out-of-order timestamp: nothing to smooth against, hold.
            return self.x_prev

        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroFilter2D:
    """Two independent OneEuroFilters, for smoothing an (x, y) cursor position."""

    def __init__(
        self,
        t0: float,
        x0: tuple[float, float],
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self._fx = OneEuroFilter(t0, x0[0], min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._fy = OneEuroFilter(t0, x0[1], min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def __call__(self, t: float, point: tuple[float, float]) -> tuple[float, float]:
        return self._fx(t, point[0]), self._fy(t, point[1])
