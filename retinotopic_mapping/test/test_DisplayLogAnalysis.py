import os
import unittest
import retinotopic_mapping.DisplayLogAnalysis as dla

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestDisplayLogAnalysis(unittest.TestCase):

    def setUp(self):
        self.log_path = os.path.join(curr_folder, 'test_data',
                                      '180312155448-CombinedStimuli-MMOUSE-USER-TEST-notTriggered-complete.pkl')
        self.log = dla.DisplayLogAnalyzer(log_path=self.log_path)

    def test_DisplayLogAnalyzer(self):

        stim_dict = self.log.get_stim_dict()
        pd_onsets_seq = self.log.analyze_photodiode_onsets_sequential(stim_dict=stim_dict, pd_thr=-0.5)
        pd_onsets_com = self.log.analyze_photodiode_onsets_combined(pd_onsets_seq=pd_onsets_seq, is_dgc_blocked=True)
        assert (len(pd_onsets_com) == 14)

    def test_DisplayLogAnalyzer_LSN(self):
        stim_dict = self.log.get_stim_dict()
        # print(stim_dict['001_LocallySparseNoiseRetinotopicMapping'].keys())
        pd_onsets_seq = self.log.analyze_photodiode_onsets_sequential(stim_dict=stim_dict, pd_thr=-0.5)
        # print('\n'.join([str(p) for p in pd_onsets_seq]))
        pd_onsets_com = self.log.analyze_photodiode_onsets_combined(pd_onsets_seq=pd_onsets_seq, is_dgc_blocked=True)

        repeat = stim_dict['006_LocallySparseNoiseRetinotopicMapping']['repeat']
        iteration = stim_dict['006_LocallySparseNoiseRetinotopicMapping']['iteration']

        lsn_dict = pd_onsets_com['006_LocallySparseNoiseRetinotopicMapping']
        # print('\n'.join(lsn_dict.keys()))
        for probe_n, probe_onset in lsn_dict.items():
            assert (len(probe_onset['global_pd_onset_ind']) == repeat * iteration)