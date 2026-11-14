import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hand_interface import (MOTOR_NAMES, SampledPositionInterface, load_config,
                            model_to_motor_order, motor_to_model_order)


class InterfaceTests(unittest.TestCase):
    def controller(self, **kwargs):
        return SampledPositionInterface(np.zeros(16), .002, .2, 400, 14, **kwargs)

    def test_motor_labels_and_independent_joint_mapping(self):
        expected = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
        self.assertEqual(model_to_motor_order(np.arange(16)).tolist(), expected)
        self.assertEqual(MOTOR_NAMES[0], "unitA-DIP")
        self.assertEqual(MOTOR_NAMES[15], "unitD-MCP-FLX")
        for model_index, motor_id in enumerate(expected):
            value = np.zeros(16); value[model_index] = -.23
            output = model_to_motor_order(value)
            self.assertEqual(output[motor_id], -.23)
            np.testing.assert_array_equal(motor_to_model_order(output), value)

    def test_hold_and_sample_rates(self):
        c = self.controller()
        for i in range(500):
            c.step(np.full(16, i*.002), np.zeros(16), np.zeros(16))
        self.assertEqual(c.counts, {"command": 50, "feedback": 20, "diagnostics": 5})
        sample = c.position_samples[1]
        self.assertAlmostEqual(sample["time_s"], .05)
        self.assertAlmostEqual(sample["command_native_rad_by_motor_id"][0], .04)
        self.assertAlmostEqual(c.command[0], .98)

    def test_local_servo_not_delayed_until_host_feedback(self):
        c = self.controller()
        c.step(np.zeros(16), np.zeros(16), np.zeros(16))
        torque = c.step(np.zeros(16), np.full(16, .01), np.zeros(16))
        np.testing.assert_array_equal(torque, np.full(16, -.2))
        self.assertEqual(c.counts["feedback"], 1)

    def test_bad_vectors_rejected_without_advancing_clock(self):
        c = self.controller()
        for bad in ([0]*15, [float("nan")]*16, [float("inf")]*16):
            with self.assertRaises(ValueError): c.step(bad, np.zeros(16), np.zeros(16))
        self.assertEqual(c.tick, 0)

    def test_invalid_sample_period_rejected(self):
        config = load_config(); config["command_interval_ms"] = 19
        with self.assertRaises(ValueError): self.controller(config=config)

    def test_bad_torque_ceiling_rejected(self):
        for ceiling in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                SampledPositionInterface(np.zeros(16), .002, ceiling, 400, 14)

    def test_report_does_not_invent_current_feedback(self):
        c = self.controller(); c.step(np.ones(16), np.zeros(16), np.zeros(16))
        report = c.report()
        self.assertEqual(report["hardware_config"]["model_number"], 1240)
        self.assertIn("simulated_torque_nm_by_motor_id", report["diagnostic_samples"][0])
        self.assertIn("no_hardware_output", report["qualification"])


if __name__ == "__main__":
    unittest.main()
