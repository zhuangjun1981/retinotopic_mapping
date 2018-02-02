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
        pd_onset_combined = dla.DisplayLogAnalyzer(log_path=self.log_path)