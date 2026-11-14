import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from speed_sweep import speed_metrics,select_profile


class SpeedTests(unittest.TestCase):
    def test_speed_uses_post_ramp_actual_time_and_requested_axis(self):
        metrics={'task':{'direction':'left'},'task_screen_pass':True,
                 'max_tracking_error_deg':2.,'max_tilt_relative_terrain_deg':4.}
        trace=[{'time_s':0.,'root_xyz_m':[0,-100,0]},
               {'time_s':3.,'root_xyz_m':[0,0,0]}, {'time_s':8.,'root_xyz_m':[.005,.1,0]}]
        result=speed_metrics(metrics,trace)
        self.assertAlmostEqual(result['post_ramp_speed_mm_s'],20.)
        self.assertAlmostEqual(result['post_ramp_cross_speed_mm_s'],1.)
        self.assertTrue(result['eligible_for_profile'])
        metrics['max_tracking_error_deg']=9.
        self.assertFalse(speed_metrics(metrics,trace)['eligible_for_profile'])

    def test_selection_ignores_faster_ineligible_candidate(self):
        rows=[]
        for direction in ('forward','backward','left','right'):
            for period,speed,eligible in [(4.,5.,True),(3.,10.,True),(2.,30.,False)]:
                rows.append({'task':{'direction':direction,'period_s':period,'stride_m':.02,
                                     'planner':'cartesian','name':direction+str(period)},
                             'speed':{'post_ramp_speed_mm_s':speed,'eligible_for_profile':eligible}})
        profile=select_profile(rows)
        self.assertTrue(all(v['period_s']==3. for v in profile['directions'].values()))
        self.assertTrue(all(v['speedup']==2. for v in profile['directions'].values()))


if __name__=='__main__':unittest.main()
