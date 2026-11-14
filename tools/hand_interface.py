"""Simulation-only position interface; no serial device or hardware output.

Vectors at the interface use motor-ID order but retain *native model radians*.
Reordering is NOT an encoder calibration or a motor direction conversion.
The plant PD gains and torque ceiling are explicit simulation assumptions.
"""
import json
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/xc330_hand.json"
MODEL_NAMES = tuple(f"unit{leg}-{joint}" for leg in "ABCD"
                    for joint in ("MCP-FLX", "MCP-ABD", "PIP", "DIP"))
MOTOR_NAMES = tuple(f"unit{leg}-{joint}" for leg in "ABCD"
                    for joint in ("DIP", "PIP", "MCP-ABD", "MCP-FLX"))
MOTOR_FROM_MODEL = np.array([MODEL_NAMES.index(name) for name in MOTOR_NAMES])
MODEL_FROM_MOTOR = np.argsort(MOTOR_FROM_MODEL)


def vector(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (16,) or not np.all(np.isfinite(values)):
        raise ValueError("expected 16 finite joint values")
    return values


def model_to_motor_order(values):
    return vector(values)[MOTOR_FROM_MODEL].copy()


def motor_to_model_order(values):
    return vector(values)[MODEL_FROM_MOTOR].copy()


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    expected = {"model": "XC330-M288-T", "model_number": 1240,
                "motor_ids": list(range(16)), "fingers": list("ABCD"),
                "motor_joint_order": ["DIP", "PIP", "MCP-ABD", "MCP-FLX"],
                "operating_mode": 5}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported hardware configuration: {key}")
    return config


class SampledPositionInterface:
    """50 Hz position requests, 20 Hz position and 5 Hz diagnostic feedback.

    The motor's local servo still observes the simulated plant every physics
    step. Host feedback sampling must not slow that local loop to 20 Hz.
    This approximates the command boundary, not XC firmware/electrical dynamics.
    """
    def __init__(self, initial_position, dt, torque_limit_nm, kp, kd, config=None,
                 rate_limits=None, joint_limits=None):
        self.config = load_config() if config is None else config
        self.dt = float(dt)
        self.torque_limit_nm = float(torque_limit_nm)
        self.kp, self.kd = float(kp), float(kd)
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.dt, self.torque_limit_nm, self.kp, self.kd)):
            raise ValueError("dt, torque ceiling and plant gains must be positive and finite")
        self.periods = {}
        for name, key in (("command", "command_interval_ms"),
                          ("feedback", "position_feedback_interval_ms"),
                          ("diagnostics", "diagnostics_interval_ms")):
            interval = float(self.config[key]) / 1000.0
            if not np.isfinite(interval) or interval <= 0:
                raise ValueError(f"invalid {key}")
            steps = round(interval / self.dt)
            if steps < 1 or not np.isclose(steps * self.dt, interval, rtol=0, atol=1e-10):
                raise ValueError(f"{key} must be an integer number of physics steps")
            self.periods[name] = steps
        self.command = model_to_motor_order(initial_position)
        self.rate_limits = rate_limits
        self.governor = None
        if rate_limits is not None:
            from trajectory import PositionGovernor
            self.governor = PositionGovernor(self.command, self.periods["command"]*self.dt,
                                              rate_limits[0], rate_limits[1])
        self.joint_limits = None if joint_limits is None else np.asarray(joint_limits,dtype=float)
        if self.joint_limits is not None:
            if (self.joint_limits.shape != (16,2) or not np.all(np.isfinite(self.joint_limits))
                    or np.any(self.joint_limits[:,0]>=self.joint_limits[:,1])):
                raise ValueError("expected 16 valid model-space joint ranges")
        self.command_samples = []
        self.tick = 0
        self.counts = {name: 0 for name in self.periods}
        self.position_samples = []
        self.diagnostic_samples = []
        self.max_tracking_error_rad = 0.0
        self.saturated_joint_steps = 0

    def step(self, requested_position, position, velocity):
        requested, position, velocity = map(vector, (requested_position, position, velocity))
        t = self.tick * self.dt
        if self.tick % self.periods["command"] == 0:
            desired = model_to_motor_order(requested)
            self.command = self.governor.update(desired) if self.governor else desired
            if self.joint_limits is not None:
                q = motor_to_model_order(self.command)
                if np.any(q<self.joint_limits[:,0]) or np.any(q>self.joint_limits[:,1]):
                    raise ValueError("shaped command crossed model joint limits")
            self.command_samples.append({"time_s": t,
                "requested_native_rad_by_motor_id": desired.tolist(),
                "command_native_rad_by_motor_id": self.command.tolist()})
            self.counts["command"] += 1
        target = motor_to_model_order(self.command)
        error = target - position
        self.max_tracking_error_rad = max(self.max_tracking_error_rad, float(np.max(np.abs(error))))
        unclipped = self.kp * error - self.kd * velocity
        self.saturated_joint_steps += int(np.count_nonzero(np.abs(unclipped) > self.torque_limit_nm))
        torque = np.clip(unclipped, -self.torque_limit_nm, self.torque_limit_nm)
        if self.tick % self.periods["feedback"] == 0:
            self.counts["feedback"] += 1
            self.position_samples.append({"time_s": t,
                "command_native_rad_by_motor_id": self.command.tolist(),
                "position_native_rad_by_motor_id": model_to_motor_order(position).tolist()})
        if self.tick % self.periods["diagnostics"] == 0:
            self.counts["diagnostics"] += 1
            self.diagnostic_samples.append({"time_s": t,
                "velocity_native_rad_s_by_motor_id": model_to_motor_order(velocity).tolist(),
                "simulated_torque_nm_by_motor_id": model_to_motor_order(torque).tolist()})
        self.tick += 1
        return torque

    def report(self):
        return {"interface": "sampled_position", "hardware_config": self.config,
                "motor_id_to_joint": dict(enumerate(MOTOR_NAMES)),
                "coordinate_space": "native_model_radians_in_motor_id_order_not_encoder_angles",
                "samples": self.counts.copy(),
                "simulation_pd_gains": {"kp": self.kp, "kd": self.kd},
                "simulation_torque_limit_nm": self.torque_limit_nm,
                "max_tracking_error_rad": self.max_tracking_error_rad,
                "torque_saturation_fraction": self.saturated_joint_steps / max(1, 16*self.tick),
                "command_samples": self.command_samples,
                "host_rate_limits": self.rate_limits,
                "position_samples": self.position_samples,
                "diagnostic_samples": self.diagnostic_samples,
                "qualification": "simulation_only_no_current_to_torque_calibration_no_hardware_output",
                "limitations": ["PD plant is not a measured XC330 firmware model",
                    "350 mA reference setting is not converted to Nm",
                    "reference PID registers and profile settings are metadata, not plant gains",
                    "joint zeros, directions and physical ROM require calibration",
                    "baseline fingertip-only collision and friction assumptions remain"]}
