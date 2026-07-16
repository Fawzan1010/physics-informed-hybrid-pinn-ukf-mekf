from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.utils.math_utils import skew
from src.utils.quaternion import quat_to_dcm
from src.utils.state import unpack_state


@dataclass
class SpacecraftParams:
    mu_earth: float = 398600.4418
    radius_earth: float = 6378.137
    j2: float = 1.08262668e-3
    mass: float = 920.0
    area: float = 12.0
    cd: float = 2.2
    cr: float = 1.3
    inertia: np.ndarray = field(default_factory=lambda: np.diag([145.0, 160.0, 120.0]))

    def __post_init__(self) -> None:
        self.inertia = np.asarray(self.inertia, dtype=float).reshape(3, 3)


@dataclass
class Environment:
    sun_vec_i: np.ndarray
    earth_vec_i: np.ndarray
    magnetic_field_i: np.ndarray
    rho_atm: float
    weather_index: float


@dataclass
class ControlInput:
    torque_cmd: np.ndarray
    accel_cmd: np.ndarray


def orbital_elements_to_state(r: np.ndarray, v: np.ndarray, params: SpacecraftParams):
    return r, v


def gravity_j2(r: np.ndarray, params: SpacecraftParams) -> np.ndarray:
    mu = params.mu_earth
    Re = params.radius_earth
    x, y, z = r
    rnorm = np.linalg.norm(r) + 1e-12
    a = -mu * r / rnorm**3
    k = 1.5 * params.j2 * mu * Re**2 / rnorm**5
    zx = z / rnorm
    f = 5 * zx * zx - 1
    a_j2 = np.array([
        x * f,
        y * f,
        z * (5 * zx * zx - 3),
    ]) * k
    return a + a_j2


def atmospheric_density(r: np.ndarray, weather_index: float, params: SpacecraftParams) -> float:
    alt = np.linalg.norm(r) - params.radius_earth
    rho0 = 3.614e-13 * (1.0 + 0.9 * weather_index)
    H = 88.0
    return float(rho0 * np.exp(-(alt - 400.0) / H))


def drag_acceleration(r: np.ndarray, v: np.ndarray, weather_index: float, params: SpacecraftParams) -> np.ndarray:
    rho = atmospheric_density(r, weather_index, params)
    vrel = v
    speed = np.linalg.norm(vrel) + 1e-12
    coeff = -0.5 * rho * params.cd * params.area / params.mass
    return coeff * speed * vrel


def srp_acceleration(r: np.ndarray, sun_vec_i: np.ndarray, weather_index: float, params: SpacecraftParams) -> np.ndarray:
    sun = sun_vec_i / (np.linalg.norm(sun_vec_i) + 1e-12)
    return 1e-7 * params.cr * (1.0 + 0.2 * weather_index) * sun


def environmental_fields(t: float, state: np.ndarray, scenario: str, params: SpacecraftParams) -> Environment:
    unpack = unpack_state(state)
    r = unpack['r']
    theta = 0.001 * t
    sun = np.array([np.cos(theta), np.sin(theta), 0.15])
    earth = -r / (np.linalg.norm(r) + 1e-12)
    magnetic = np.array([2e-5 * np.cos(0.2 * t), 2e-5 * np.sin(0.2 * t), 1e-5])
    weather = 0.1
    if scenario == 'storm':
        weather = 1.0 + 0.6 * np.sin(0.05 * t)
    elif scenario == 'high_disturbance':
        weather = 0.7
    elif scenario == 'degraded_measurements':
        weather = 0.3
    return Environment(sun, earth, magnetic, atmospheric_density(r, weather, params), weather)


def control_profile(t: float, scenario: str) -> ControlInput:
    torque = np.array([2e-4 * np.sin(0.03 * t), 1.5e-4 * np.cos(0.021 * t), 1e-4 * np.sin(0.017 * t)])
    accel = np.array([1e-6 * np.sin(0.01 * t), 1e-6 * np.cos(0.013 * t), 1e-6 * np.sin(0.02 * t)])
    if scenario == 'maneuver':
        burst = 1.0 if 60 <= t <= 90 or 150 <= t <= 165 else 0.0
        torque += burst * np.array([3.0e-3, -2.0e-3, 1.5e-3])
        accel += burst * np.array([5e-5, -4e-5, 3e-5])
    return ControlInput(torque, accel)


def state_derivative(t: float, x: np.ndarray, control: ControlInput, env: Environment, params: SpacecraftParams) -> np.ndarray:
    d = unpack_state(x)
    q = d['q']
    w = d['w']
    r = d['r']
    v = d['v']
    bg = d['bg']
    ba = d['ba']
    td = d['td']
    ad = d['ad']

    qmat = np.array([
        [0.0, -w[0], -w[1], -w[2]],
        [w[0], 0.0, w[2], -w[1]],
        [w[1], -w[2], 0.0, w[0]],
        [w[2], w[1], -w[0], 0.0],
    ])
    qdot = 0.5 * qmat @ q
    I = params.inertia
    wdot = np.linalg.solve(I, control.torque_cmd + td - np.cross(w, I @ w))
    rdot = v
    a_grav = gravity_j2(r, params)
    a_drag = drag_acceleration(r, v, env.weather_index, params)
    a_srp = srp_acceleration(r, env.sun_vec_i, env.weather_index, params)
    vdot = a_grav + a_drag + a_srp + control.accel_cmd + ad
    bgdot = -2e-4 * bg
    badot = -2e-4 * ba
    tddot = -0.02 * td
    addot = -0.02 * ad
    return np.hstack([qdot, wdot, rdot, vdot, bgdot, badot, tddot, addot])


def rk4_step(x: np.ndarray, t: float, dt: float, control: ControlInput, env: Environment, params: SpacecraftParams) -> np.ndarray:
    k1 = state_derivative(t, x, control, env, params)
    k2 = state_derivative(t + 0.5 * dt, x + 0.5 * dt * k1, control, env, params)
    k3 = state_derivative(t + 0.5 * dt, x + 0.5 * dt * k2, control, env, params)
    k4 = state_derivative(t + dt, x + dt * k3, control, env, params)
    xn = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    xn[:4] = xn[:4] / (np.linalg.norm(xn[:4]) + 1e-12)
    return xn
