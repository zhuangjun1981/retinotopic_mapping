import os
import unittest
import retinotopic_mapping.StimulusRoutines as sr
import retinotopic_mapping.MonitorSetup as ms

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestSimulation(unittest.TestCase):

    def setUp(self):

        self._monitor = ms.Monitor(resolution=(1200,1600), dis=15., mon_width_cm=40., mon_height_cm=30., C2T_cm=15.,
                                   C2A_cm=20., mon_tilt=30., downsample_rate=10)
        self._indicator = ms.Indicator(self._monitor)

    def test_UniformContrast_generate_movie_by_index(self):
        uc = sr.UniformContrast(monitor=self._monitor, indicator=self._indicator, duration=0.1, color=1.,
                                pregap_dur=1., postgap_dur=1.5, background=0., coordinate='degree')
        full_sequence, full_dict = uc.generate_movie_by_index()

        assert (full_sequence.shape == (3, 120, 160))
        assert (len(full_dict['stimulation']['index_to_display']) == 156)

        frames = full_dict['stimulation']['frames']
        all_frames = []
        for ind in full_dict['stimulation']['index_to_display']:
            all_frames.append(frames[ind])

        # print (all_frames)

        for i in range(60):
            assert (all_frames[i] == (0., -1.))

        for i in range(60, 66):
            assert (all_frames[i] == (1., 1.))

        for i in range(66, 156):
            assert (all_frames[i] == (0., -1.))

    def test_FlashingCircle_generate_movie_by_index(self):
        fc = sr.FlashingCircle(monitor=self._monitor, indicator=self._indicator, center=(90., 0.), flash_frame=30,
                               color=-1., pregap_dur=0.5, postgap_dur=1.2, background=1., coordinate='degree')
        full_sequence, full_dict = fc.generate_movie_by_index()

        assert (full_sequence.shape == (3, 120, 160))
        assert (len(full_dict['stimulation']['index_to_display']) == 132)

        frames = full_dict['stimulation']['frames']
        all_frames = []
        for ind in full_dict['stimulation']['index_to_display']:
            all_frames.append(frames[ind])

        # print (all_frames)

        for i in range(30):
            assert (all_frames[i] == (1., -1.))

        for i in range(30, 60):
            assert (all_frames[i] == (1., 1.))

        for i in range(60, 132):
            assert (all_frames[i] == (0., -1.))

    def test_SparseNoise_generate_movie_by_index(self):
        pass

    def test_DriftingGratingCircle_generate_movie_by_index(self):
        pass