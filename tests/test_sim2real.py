import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from trajectory import PositionGovernor,check_trajectory,EncoderCalibration
from hand_interface import MOTOR_NAMES


def calibration_fixture():
    return {'status':'measured','measurement_reference':'synthetic test fixture only',
            'model':'XC330-M288-T','model_number':1240,'supply_voltage_v':5.0,
            'joints':[{'motor_id':i,'model_joint':name,'native_to_encoder_sign':(-1 if i%2 else 1),
                       'encoder_at_native_zero_rad':3.14,'native_lower_rad':-2.,
                       'native_upper_rad':2.,'drive_mode_raw':0} for i,name in enumerate(MOTOR_NAMES)]}


class TrajectoryTests(unittest.TestCase):
    def test_limits_through_reversals(self):
        g=PositionGovernor(np.zeros(16),.02,2,8);q=[g.position.copy()]
        for i in range(1,500):
            goal=1. if i<100 else (-1. if i<300 else .4)
            q.append(g.update(np.full(16,goal)))
        result=check_trajectory(np.arange(500)*.02,q,2,8)
        self.assertLessEqual(result['peak_command_acceleration_rad_s2'],8+1e-6)
        np.testing.assert_allclose(q[-1],.4,atol=1e-5)

    def test_nan_and_rate_violation_rejected(self):
        with self.assertRaises(ValueError):PositionGovernor(np.zeros(16),.02,float('nan'),8)
        q=np.zeros((5,16));q[1:,0]=1
        with self.assertRaises(ValueError):check_trajectory(np.arange(5)*.02,q,2,8)
        with self.assertRaises(ValueError):check_trajectory([0,.02,.05],np.zeros((3,16)),2,8)

    def test_encoder_sign_and_zero_with_quantization(self):
        c=EncoderCalibration(calibration_fixture());q=np.tile(np.linspace(-.5,.5,16),(3,1))
        pulse=c.encode(q)
        actual=(pulse*(2*np.pi/4096)-c.zero)*c.sign
        np.testing.assert_allclose(actual,q,atol=np.pi/4096)

    def test_unmeasured_and_wrong_model_rejected(self):
        for key,value in [('status','unmeasured'),('model_number',1200),('supply_voltage_v',None)]:
            c=calibration_fixture();c[key]=value
            with self.assertRaises(ValueError):EncoderCalibration(c)
        c=calibration_fixture();c['joints'][0]['native_to_encoder_sign']=None
        with self.assertRaises(ValueError):EncoderCalibration(c)

    def test_calibrated_rom_rejects_instead_of_clipping(self):
        c=EncoderCalibration(calibration_fixture())
        with self.assertRaises(ValueError):c.encode(np.full((3,16),2.1))


if __name__=='__main__':unittest.main()
