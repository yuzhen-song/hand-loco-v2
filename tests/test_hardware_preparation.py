import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_hardware import read_snapshot,REGISTERS
from trajectory import EncoderCalibration
from hand_interface import load_config
import json


class ReadOnlyPacket:
    def __init__(self,wrong_model=False,error=False):
        self.calls=[];self.wrong_model=wrong_model;self.error=error
    def __getattr__(self,name):
        if name not in ('read1ByteTxRx','read2ByteTxRx','read4ByteTxRx'):
            raise AssertionError('unexpected hardware operation '+name)
        def read(port,motor,address):
            self.calls.append((name,motor,address))
            value={0:1200 if self.wrong_model else 1240,7:motor,144:50,132:0xffffffff,126:0xffff}.get(address,0)
            return value,0,1 if self.error else 0
        return read


class HardwarePreparationTests(unittest.TestCase):
    def test_audit_reads_only_and_decodes_signed_feedback(self):
        packet=ReadOnlyPacket();result=read_snapshot(object(),packet)
        self.assertEqual(len(packet.calls),16*len(REGISTERS))
        self.assertEqual(result['motors'][0]['input_voltage_v'],5.)
        self.assertEqual(result['motors'][0]['signed_position_pulses'],-1)
        self.assertEqual(result['motors'][0]['present_current_ma'],-1)
    def test_wrong_model_and_device_error_stop_audit(self):
        with self.assertRaises(ValueError):read_snapshot(object(),ReadOnlyPacket(wrong_model=True))
        with self.assertRaises(RuntimeError):read_snapshot(object(),ReadOnlyPacket(error=True))
    def test_shipped_template_cannot_be_used_as_calibration(self):
        root=Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            EncoderCalibration(json.loads((root/'configs/hardware_calibration.example.json').read_text()))
        self.assertIsNone(load_config()['hardware_calibration']['supply_voltage_v'])


class OfflinePlanTests(unittest.TestCase):
    def test_measured_synthetic_fixture_still_is_not_motion_ready(self):
        import tempfile
        from prepare_hardware import prepare
        from test_sim2real import calibration_fixture
        from hand_interface import MOTOR_NAMES
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='.offline_test_',dir=root) as temp:
            path=Path(temp)
            (path/'calibration.json').write_text(json.dumps(calibration_fixture()))
            (path/'metrics.json').write_text(json.dumps({'screen_pass':True,'stop_reason':'duration_completed'}))
            data={'motor_id_to_joint':{str(i):n for i,n in enumerate(MOTOR_NAMES)},
                  'coordinate_space':'native_model_radians_in_motor_id_order_not_encoder_angles',
                  'hardware_config':load_config(),'host_rate_limits':[2,8],
                  'command_samples':[{'time_s':i*.02,'command_native_rad_by_motor_id':[0.]*16} for i in range(3)]}
            (path/'interface.json').write_text(json.dumps(data))
            result=prepare(path,path/'calibration.json')
            self.assertFalse(result['hardware_motion_ready'])
            self.assertEqual(len(result['goal_position_pulses']),3)


if __name__=='__main__':unittest.main()
