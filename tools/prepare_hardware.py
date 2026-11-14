"""Offline encoder-plan validation. Never connects to or moves a robot."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from trajectory import EncoderCalibration,check_trajectory
from hand_interface import MOTOR_NAMES,load_config


def prepare(run_directory, calibration_path):
    run=Path(run_directory);calibration_path=Path(calibration_path)
    c=EncoderCalibration(json.loads(calibration_path.read_text()))
    metrics=json.loads((run/'metrics.json').read_text())
    interface=json.loads((run/'interface.json').read_text())
    if metrics.get('stop_reason')!='duration_completed' or not metrics.get('screen_pass'):
        raise ValueError('simulation candidate did not complete its exploratory screen')
    expected={str(i):n for i,n in enumerate(MOTOR_NAMES)}
    if interface.get('motor_id_to_joint')!=expected or interface.get('coordinate_space')!='native_model_radians_in_motor_id_order_not_encoder_angles':
        raise ValueError('wrong source coordinate convention or joint order')
    config=load_config()
    if interface['hardware_config']['model_number']!=config['model_number']:
        raise ValueError('source actuator model mismatch')
    records=interface['command_samples'];times=[r['time_s'] for r in records]
    q=np.array([r['command_native_rad_by_motor_id'] for r in records])
    rates=interface.get('host_rate_limits')
    if rates is None:raise ValueError('trajectory has no validated command-rate limits')
    bounds=check_trajectory(times,q,*rates)
    pulses=c.encode(q)
    reconstructed=(pulses*(2*np.pi/4096)-c.zero)*c.sign
    # Quantization can add finite-difference acceleration. Report it explicitly;
    # this offline file is not a hardware-ready trajectory or execution backend.
    v=np.diff(reconstructed,axis=0)/.02
    a=np.diff(np.vstack([np.zeros((1,16)),v]),axis=0)/.02
    quantized_bounds={'peak_velocity_rad_s':float(np.max(np.abs(v))),
                      'peak_acceleration_rad_s2':float(np.max(np.abs(a)))}
    return {'status':'offline_encoded_candidate_not_authorized_for_motion',
            'hardware_motion_ready':False,'motor_ids':list(range(16)),
            'model':'XC330-M288-T','model_number':1240,'period_s':.02,
            'required_drive_modes':c.drive_modes,'times_s':times,
            'goal_position_pulses':pulses.tolist(),'native_command_bounds':bounds,
            'quantized_command_bounds':quantized_bounds,
            'quantized_rate_limits_pass':bool(quantized_bounds['peak_velocity_rad_s']<=rates[0]+1e-7 and
                                              quantized_bounds['peak_acceleration_rad_s2']<=rates[1]+1e-6),
            'source_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (run/'metrics.json',run/'interface.json',calibration_path)},
            'remaining_checks':['fresh device readback, including drive mode, firmware and voltage',
                'initial encoder pose and supported commissioning procedure',
                'quantized command smoothing and replay in simulation if rate checks fail',
                'measured torque/servo response and qualified self-collision geometry',
                'live feedback freshness, tracking/temperature faults and stop behavior']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output exists')
    try:result=prepare(args.run,args.calibration)
    except (ValueError,KeyError,TypeError) as e:p.error(str(e))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Offline plan saved; no hardware communication:',args.output)


if __name__=='__main__':main()
