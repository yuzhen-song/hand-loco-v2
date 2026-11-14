"""Explicit read-only XC330 device audit. No write/enable/position instructions.

Run only on the computer physically connected to the hand, with other serial
clients stopped. Requires the optional Dynamixel SDK; importing this module
never opens a device. A result is a snapshot, not permission to execute motion.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

REGISTERS={
 'model_number':(0,2),'firmware':(6,1),'id':(7,1),'drive_mode':(10,1),
 'operating_mode':(11,1),'current_limit_raw':(38,2),'torque_enable':(64,1),
 'hardware_error':(70,1),'position_d_gain':(80,2),'position_i_gain':(82,2),
 'position_p_gain':(84,2),'bus_watchdog_raw':(98,1),'goal_current_raw':(102,2),
 'profile_acceleration_raw':(108,4),'profile_velocity_raw':(112,4),
 'present_current_raw':(126,2),'present_position_raw':(132,4),
 'present_input_voltage_raw':(144,2),'temperature_c':(146,1)}


def read_snapshot(port,packet,comm_success=0):
    motors=[]
    for motor in range(16):
        row={'requested_id':motor}
        for name,(address,width) in REGISTERS.items():
            reader=getattr(packet,f'read{width}ByteTxRx')
            value,result,error=reader(port,motor,address)
            if result!=comm_success or error:
                raise RuntimeError(f'read failed: ID {motor}, {name}, communication={result}, device={error}')
            row[name]=int(value)
            if name=='model_number' and value!=1240:
                raise ValueError(f'ID {motor}: model {value}, expected XC330-M288-T / 1240')
        if row['id']!=motor:raise ValueError('returned motor ID mismatch')
        row['input_voltage_v']=row['present_input_voltage_raw']*.1
        raw=row['present_position_raw'];row['signed_position_pulses']=raw-(1<<32) if raw&(1<<31) else raw
        raw=row['present_current_raw'];row['present_current_ma']=raw-(1<<16) if raw&(1<<15) else raw
        motors.append(row)
    return {'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'operation':'read_only_no_hardware_writes','motors':motors}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True,help='explicit serial device on the connected computer')
    p.add_argument('--baud',type=int,default=1000000)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output exists')
    if args.baud<=0:p.error('invalid baud')
    try:
        from dynamixel_sdk import PortHandler,PacketHandler,COMM_SUCCESS
    except ImportError:
        p.error('install optional requirements-hardware.txt on the connected computer first')
    port=PortHandler(args.port)
    try:
        if not port.openPort():raise RuntimeError('cannot open explicitly selected serial device')
        if not port.setBaudRate(args.baud):raise RuntimeError('cannot set host serial baud')
        start=time.monotonic();report=read_snapshot(port,PacketHandler(2.0),COMM_SUCCESS)
        report['elapsed_s']=time.monotonic()-start
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    finally:
        port.closePort()
    print('Saved read-only snapshot:',args.output)


if __name__=='__main__':main()
