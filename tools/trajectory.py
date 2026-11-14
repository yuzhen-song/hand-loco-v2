"""Shared rate-bounded joint commands and explicit encoder calibration.

All inputs use motor-ID order. Model-native radians are never silently treated
as encoder radians. This module has no hardware I/O.
"""
import numpy as np
from hand_interface import vector


class PositionGovernor:
    """Discrete velocity/acceleration-limited reference, updated at command rate.

    It brakes using stopping distance. A moving/reversing target can be overshot;
    acceleration is never violated by snapping to the target. The caller must
    reject a reference crossing the permitted model/mechanical range.
    This is host trajectory shaping, not an emulation of servo firmware.
    """
    def __init__(self, initial, dt, velocity_limit, acceleration_limit):
        self.position = vector(initial).copy()
        self.velocity = np.zeros(16)
        self.dt, self.vmax, self.amax = map(float, (dt, velocity_limit, acceleration_limit))
        if not all(np.isfinite(x) and x > 0 for x in (self.dt, self.vmax, self.amax)):
            raise ValueError("positive finite rate limits and dt required")

    def update(self, goal):
        error = vector(goal) - self.position
        dv = self.amax * self.dt
        stop_speed = np.sqrt(dv*dv + 2*self.amax*np.abs(error)) - dv
        desired_velocity = np.sign(error) * np.minimum(self.vmax, stop_speed)
        new_velocity = np.clip(desired_velocity, self.velocity-dv, self.velocity+dv)
        self.position += new_velocity * self.dt
        self.velocity = new_velocity
        return self.position.copy()


def check_trajectory(times, positions, velocity_limit, acceleration_limit, dt=.02):
    """Validate the actual emitted commands, not the requested planner curve."""
    t = np.asarray(times, dtype=float)
    q = np.asarray(positions, dtype=float)
    if t.ndim != 1 or len(t) < 3 or q.shape != (len(t),16):
        raise ValueError("need at least three timestamped 16-joint commands")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite trajectory")
    if abs(t[0]) > 1e-9 or not np.allclose(np.diff(t),dt,rtol=0,atol=1e-9):
        raise ValueError("commands must start at zero on the 20 ms grid")
    v = np.diff(q,axis=0)/dt
    # Governor starts at rest; include acceleration into the first interval.
    a = np.diff(np.vstack([np.zeros((1,16)),v]),axis=0)/dt
    peak_v, peak_a = float(np.max(np.abs(v))),float(np.max(np.abs(a)))
    if peak_v > velocity_limit+1e-7 or peak_a > acceleration_limit+1e-6:
        raise ValueError(f"command rate violation: {peak_v=}, {peak_a=}")
    return {"peak_command_velocity_rad_s":peak_v,"peak_command_acceleration_rad_s2":peak_a}


class EncoderCalibration:
    """Measured calibration for this assembly; deliberately no reference defaults."""
    def __init__(self, data):
        if data.get('status') != 'measured' or not data.get('measurement_reference'):
            raise ValueError('calibration must identify measurements of this assembly')
        if data.get('model') != 'XC330-M288-T' or data.get('model_number') != 1240:
            raise ValueError('wrong actuator model')
        voltage = data.get('supply_voltage_v')
        if not isinstance(voltage,(int,float)) or not np.isfinite(voltage) or not 3.7 <= voltage <= 6.0:
            raise ValueError('measured XC330-M288 supply voltage required (3.7–6.0 V)')
        entries = data.get('joints',[])
        if [x.get('motor_id') for x in entries] != list(range(16)):
            raise ValueError('calibration requires exactly motor IDs 0–15 in order')
        from hand_interface import MOTOR_NAMES
        if [x.get('model_joint') for x in entries] != list(MOTOR_NAMES):
            raise ValueError('calibration joint names do not match motor IDs')
        def values(key):
            if any(x.get(key) is None for x in entries):
                raise ValueError('unmeasured calibration field: '+key)
            return vector([x[key] for x in entries])
        self.sign = values('native_to_encoder_sign')
        self.zero = values('encoder_at_native_zero_rad')
        self.lower = values('native_lower_rad')
        self.upper = values('native_upper_rad')
        if not np.all(np.isin(self.sign,[-1,1])) or np.any(self.lower>=self.upper):
            raise ValueError('invalid directions or calibrated joint ranges')
        # Drive Mode reverse bit must match the state used during calibration.
        self.drive_modes = [x.get('drive_mode_raw') for x in entries]
        if any(type(x) is not int or x not in (0,1,4,5) for x in self.drive_modes):
            raise ValueError('measured Drive Mode required; implicit torque-on modes unsupported')
        self.data = data

    def encode(self, positions):
        q=np.asarray(positions,dtype=float)
        if q.ndim != 2 or q.shape[1]!=16 or not np.all(np.isfinite(q)):
            raise ValueError('expected finite Nx16 native positions')
        if np.any(q<self.lower-1e-9) or np.any(q>self.upper+1e-9):
            raise ValueError('trajectory exceeds calibrated mechanical ROM')
        encoder=q*self.sign+self.zero
        pulses=np.rint(encoder*(4096/(2*np.pi)))
        if np.any(np.abs(pulses)>1048575):
            raise ValueError('position exceeds multi-turn register range')
        # Quantization must not put the actual commanded angle beyond the ROM.
        reconstructed=(pulses*(2*np.pi/4096)-self.zero)*self.sign
        if np.any(reconstructed<self.lower-1e-9) or np.any(reconstructed>self.upper+1e-9):
            raise ValueError('encoder quantization crosses a mechanical limit')
        return pulses.astype(np.int64)
