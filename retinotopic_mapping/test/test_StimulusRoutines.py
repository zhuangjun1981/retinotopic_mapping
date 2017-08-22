import os
import unittest
import retinotopic_mapping.StimulusRoutines as sr
import retinotopic_mapping.MonitorSetup as ms

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestSimulation(unittest.TestCase):

    def setUp(self):
        
        # Setup monitor/indicator objects
        self.monitor = ms.Monitor(resolution=(1200,1600), dis=15., 
                                   mon_width_cm=40., mon_height_cm=30., 
                                   C2T_cm=15.,C2A_cm=20., mon_tilt=30., downsample_rate=10)
        self.indicator = ms.Indicator(self.monitor)
        
        # Setup Uniform Contrast Objects
        self.UC = sr.UniformContrast(monitor=self.monitor, 
                                     indicator=self.indicator, 
                                     duration=0.1, color=1.,
                                     pregap_dur=1., postgap_dur=1.5, 
                                     background=0., coordinate='degree')
        
        self.UC_full_seq, self.UC_full_dict = self.UC.generate_movie_by_index()
        
        # Setup Flashing Circle Objects
        self.FC = sr.FlashingCircle(monitor=self.monitor, 
                                    indicator=self.indicator, 
                                    center=(90., 0.), flash_frame=30,
                                    color=-1., pregap_dur=0.5, postgap_dur=1.2, 
                                    background=1., coordinate='degree')
        
        self.FC_full_seq, self.FC_full_dict = self.FC.generate_movie_by_index()


    # UNIFORM CONTRAST TESTS
    # ======================
    def test_UC_by_index_correct_sequence_shape(self):
        
        assert (self.UC_full_seq.shape == (3, 120, 160))
        
    def test_UC_by_index_correct_number_of_unique_frames(self):
    
        
        assert (len(UC_full_dict['stimulation']['index_to_display']) == 156)


    def foo(bar):
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

    # FLASHING CIRCLE TESTS #
    # ===================== #
    def test_FlashingCircle_generate_movie_by_index(self):
        
        assert (self.FC_full_seq.shape == (3, 120, 160))
#        assert (len(full_dict['stimulation']['index_to_display']) == 132)
#
#        frames = full_dict['stimulation']['frames']
#        all_frames = []
#        for ind in full_dict['stimulation']['index_to_display']:
#            all_frames.append(frames[ind])
#
#        # print (all_frames)
#
#        for i in range(30):
#            assert (all_frames[i] == (1., -1.))
#
#        for i in range(30, 60):
#            assert (all_frames[i] == (1., 1.))
#
#        for i in range(60, 132):
#            assert (all_frames[i] == (0., -1.))

    def test_SparseNoise_generate_movie_by_index(self):
        pass

    def test_DriftingGratingCircle_generate_movie_by_index(self):
        pass
    
    
if __name__ == '__main__':
    unittest.main(verbosity=2.)