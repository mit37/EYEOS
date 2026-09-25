"""One Euro Filter tested against the reference algorithm's known behavior."""

from __future__ import annotations

import math

import pytest

from eyeos.filtering import OneEuroFilter, OneEuroFilter2D, exponential_smoothing, smoothing_factor


def test_smoothing_factor_exact_value_at_cutoff_one_over_two_pi():
    # r = 2*pi*cutoff*t_e = 2*pi*(1/(2*pi))*1 = 1  =>  alpha = 1/(1+1) = 0.5
    alpha = smoothing_factor(t_e=1.0, cutoff=1.0 / (2 * math.pi))
    assert alpha == pytest.approx(0.5)


def test_smoothing_factor_approaches_one_for_large_dt_or_cutoff():
    assert smoothing_factor(t_e=1000.0, cutoff=10.0) == pytest.approx(1.0, abs=1e-3)


def test_smoothing_factor_approaches_zero_for_tiny_dt():
    assert smoothing_factor(t_e=1e-6, cutoff=1.0) == pytest.approx(0.0, abs=1e-3)


def test_exponential_smoothing_matches_formula():
    assert exponential_smoothing(0.3, x=10.0, x_prev=0.0) == pytest.approx(3.0)
    assert exponential_smoothing(1.0, x=10.0, x_prev=999.0) == pytest.approx(10.0)
    assert exponential_smoothing(0.0, x=10.0, x_prev=999.0) == pytest.approx(999.0)


def test_constant_signal_is_returned_unchanged_at_steady_state():
    f = OneEuroFilter(t0=0.0, x0=5.0, min_cutoff=1.0, beta=0.0)
    for t in range(1, 20):
        out = f(t=float(t), x=5.0)
    assert out == pytest.approx(5.0)


def test_step_response_lags_behind_the_step():
    f = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)
    out = f(t=1.0 / 30.0, x=100.0)
    assert 0.0 < out < 100.0


def test_step_response_converges_to_the_new_value():
    f = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)
    out = 0.0
    t = 0.0
    for _ in range(300):
        t += 1.0 / 30.0
        out = f(t=t, x=100.0)
    assert out == pytest.approx(100.0, abs=0.5)


def test_lower_min_cutoff_smooths_more_on_the_same_step():
    smooth = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=0.1, beta=0.0)
    responsive = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=10.0, beta=0.0)
    out_smooth = smooth(t=1.0 / 30.0, x=100.0)
    out_responsive = responsive(t=1.0 / 30.0, x=100.0)
    assert out_smooth < out_responsive


def test_higher_beta_reduces_lag_during_fast_motion():
    # Feed a fast ramp; beta should let the high-speed derivative widen the cutoff
    # and track the ramp more closely than beta=0 does.
    no_beta = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)
    with_beta = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=5.0)
    t = 0.0
    out_no_beta = out_with_beta = 0.0
    for i in range(1, 30):
        t = i / 30.0
        x = 1000.0 * t  # fast ramp
        out_no_beta = no_beta(t=t, x=x)
        out_with_beta = with_beta(t=t, x=x)
    lag_no_beta = 1000.0 * t - out_no_beta
    lag_with_beta = 1000.0 * t - out_with_beta
    assert lag_with_beta < lag_no_beta


def test_non_advancing_timestamp_holds_previous_value():
    f = OneEuroFilter(t0=0.0, x0=5.0, min_cutoff=1.0, beta=0.0)
    out1 = f(t=1.0, x=5.0)
    out2 = f(t=1.0, x=999.0)  # same timestamp again: dt <= 0
    assert out2 == out1


def test_out_of_order_timestamp_holds_previous_value():
    f = OneEuroFilter(t0=1.0, x0=5.0, min_cutoff=1.0, beta=0.0)
    out = f(t=0.5, x=999.0)
    assert out == pytest.approx(5.0)


def test_2d_filter_smooths_each_axis_independently():
    f2d = OneEuroFilter2D(t0=0.0, x0=(0.0, 0.0), min_cutoff=1.0, beta=0.0)
    fx = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)
    fy = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)
    out2d = f2d(t=1.0 / 30.0, point=(100.0, -50.0))
    outx = fx(t=1.0 / 30.0, x=100.0)
    outy = fy(t=1.0 / 30.0, x=-50.0)
    assert out2d == (pytest.approx(outx), pytest.approx(outy))
