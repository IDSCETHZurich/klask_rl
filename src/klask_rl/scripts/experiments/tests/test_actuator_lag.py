"""CPU tests of the parametric actuator surrogate (actuator_lag.py) and the v2-like step response."""

import importlib.util
import math
from pathlib import Path

import torch

_P = Path(__file__).resolve().parents[3] / "source/klask_rl/klask_rl/tasks/manager_based/klask_rl/actuator_model/actuator_lag.py"


def _module():
    spec = importlib.util.spec_from_file_location("actuator_lag", _P)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError:  # gymnasium missing outside the container: test the filter only
        src = _P.read_text().replace("from gymnasium import Wrapper", "Wrapper = object")
        exec(compile(src, str(_P), "exec"), mod.__dict__)
    return mod


def rate_limit(seq, max_dv):
    out, v = [], 0.0
    for u in seq:
        v = v + max(-max_dv, min(max_dv, u - v))
        out.append(v)
    return out


def test_coefficients_and_limits():
    m = _module()
    assert m.lag_coefficients(0.02, 0.0, 0.0) == (0.0, 1.0)  # no lag: passes the command through
    f, a = m.lag_coefficients(0.02, 0.005, 0.01)
    assert abs(f - 0.25) < 1e-9 and abs(a - (1 - math.exp(-2))) < 1e-9
    try:
        m.lag_coefficients(0.02, 0.03, 0.01)
    except ValueError:
        pass
    else:
        raise AssertionError("dead time longer than a step must be rejected")


def test_filter_steady_state_and_reset():
    m = _module()
    lag = m.ActuatorLagStep(3, 2, 0.02, 0.005, 0.01)
    u = torch.full((3, 2), 0.7)
    for _ in range(20):
        v = lag(u)
    assert torch.allclose(v, u, atol=1e-4)  # converges to the command
    lag.reset(torch.tensor([True, False, False]))
    assert float(lag.v[0].abs().sum()) == 0 and float(lag.prev_u[0].abs().sum()) == 0
    assert torch.allclose(lag.v[1:], u[1:], atol=1e-4)


def step_response(dt=0.02, vmax=1.4, amax=35.0, dead=0.005, tau=0.010, n=12):
    m = _module()
    lag = m.ActuatorLagStep(1, 1, dt, dead, tau)
    cmd = rate_limit([vmax] * n, amax * dt)  # VelocityScaleWrapper applies the cap before the lag
    return [float(lag(torch.tensor([[c]]))[0, 0]) for c in cmd]


def test_v2_step_response_fast():
    out = step_response()
    t90 = next(i for i, v in enumerate(out) if v >= 0.9 * 1.4)
    assert t90 <= 3, out  # within 60 ms (current learned model: ~180 ms to 90% of 0.4 m/s)


if __name__ == "__main__":
    test_coefficients_and_limits()
    test_filter_steady_state_and_reset()
    test_v2_step_response_fast()
    for dead, tau in ((0.005, 0.010), (0.010, 0.015), (0.0, 0.0)):
        out = step_response(dead=dead, tau=tau)
        times = {p: next((i + 1) * 20 for i, v in enumerate(out) if v >= p * 1.4) for p in (0.1, 0.5, 0.9)}
        print(f"dead {dead*1000:.0f} ms, tau {tau*1000:.0f} ms, 35 m/s^2, 1.4 m/s: " + " ".join(f"{v:.2f}" for v in out[:6])
              + f" | 10/50/90% after {times[0.1]}/{times[0.5]}/{times[0.9]} ms")
    print("all tests passed")
