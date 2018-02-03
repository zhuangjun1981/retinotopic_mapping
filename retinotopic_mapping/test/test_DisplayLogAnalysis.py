import os
import unittest
import retinotopic_mapping.DisplayLogAnalysis as dla

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestDisplayLogAnalysis(unittest.TestCase):

    def setUp(self):
        self.log_path = os.path.join(curr_folder, 'test_data',
                                     '180202152841-CombinedStimuli-MMOUSE-USER-TEST-notTriggered-complete.pkl')

    def test_DisplayLogAnalyzer(self):
        log = dla.DisplayLogAnalyzer(log_path=self.log_path)
        stim_dict = log.get_stim_dict()
        pd_onsets_seq = log.analyze_photodiode_onsets_sequential(stim_dict=stim_dict, pd_thr=-0.5)
        pd_onsets_com = log.analyze_photodiode_onsets_combined(pd_onsets_seq=pd_onsets_seq, is_dgc_blocked=True)
        assert (len(pd_onsets_com) == 14)