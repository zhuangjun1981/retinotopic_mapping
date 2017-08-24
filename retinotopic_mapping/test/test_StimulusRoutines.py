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
        
        # Setup Sparse Noise Objects
        self.SN = sr.SparseNoise(monitor=self.monitor,
                                 indicator=self.indicator,
                                 probe_frame_num=10)
        self.SN_full_seq, self.SN_full_dict = self.SN.generate_movie_by_index()
        
        # Setup Drifting Grating Circle objects
        self.DGC = sr.DriftingGratingCircle(monitor=self.monitor,
                                            indicator=self.indicator,
                                            sf_list=(0.08,),
                                            tf_list=(4.0,),
                                            dire_list=(0.,),
                                            con_list=(1.,),
                                            size_list=(10.,))
        self.DGC_full_seq, self.DGC_full_dict = self.DGC.generate_movie_by_index()

    # UNIFORM CONTRAST TESTS
    # ======================
    def test_UC_by_index_correct_sequence_shape(self):
        assert (self.UC_full_seq.shape == (3, 120, 160))
        
    def test_UC_by_index_correct_number_of_unique_frames(self):
        assert (len(self.UC_full_dict['stimulation']['index_to_display']) == 156)

    def test_UC_index_frames_are_correct(self):
        frames = self.UC_full_dict['stimulation']['frames']
        all_frames = []
        for ind in self.UC_full_dict['stimulation']['index_to_display']:
            all_frames.append(frames[ind])

        # Parameters defining where the frame blocks should start and end
        ref_rate = self.monitor.refresh_rate
        pregap_end = self.UC.pregap_frame_num
        on_end = pregap_end + int(self.UC.duration*ref_rate)
        postgap_end = on_end + self.UC.postgap_frame_num
        
        for i in range(pregap_end):
            assert (all_frames[i] == (0., -1.))

        for i in range(pregap_end, on_end):
            assert (all_frames[i] == (1., 1.))

        for i in range(on_end, postgap_end):
            assert (all_frames[i] == (0., -1.))

    # FLASHING CIRCLE TESTS #
    # ===================== #
    def test_FC_by_index_correct_sequence_shape(self):
        assert (self.FC_full_seq.shape == (4, 120, 160))
        
    def test_FC_by_index_correct_number_of_unique_frames(self):
        assert (len(self.FC_full_dict['stimulation']['index_to_display']) == 132)

    def test_FC_index_frames_are_correct(self):
        frames = self.FC_full_dict['stimulation']['frames']
        all_frames = []
        for ind in self.FC_full_dict['stimulation']['index_to_display']:
            all_frames.append(frames[ind])
        
        # Parameters defining where the frame blocks should start and end
        pregap_end = self.FC.pregap_frame_num
        flash_frames= self.FC.flash_frame
        flashing_end = pregap_end + flash_frames
        postgap_end = flashing_end + self.FC.postgap_frame_num
        
        for i in range(1):
            assert (all_frames[i] == (0., 1., -1.))

        for i in range(1,pregap_end):
            assert (all_frames[i] == (0., 0., -1.))
        
        # Flashing frames
        for i in range(pregap_end,flashing_end):
            assert (all_frames[i] == (1.,0.,1.))

        for i in range(flashing_end, postgap_end):
            assert (all_frames[i] == (0., 0., -1.))
    
    # SPARSE NOISE TESTS #
    # ================== #
    def test_SN_by_index_correct_sequence_shape(self):
        stim_on_frames = len(self.SN._generate_grid_points_sequence())*self.SN.probe_frame_num
        index_frames = self.SN.pregap_frame_num + stim_on_frames + self.SN.postgap_frame_num
        
        assert (len(self.SN_full_dict['stimulation']['index_to_display']) == index_frames)

    # DRIFTING GRATING CIRCLE TESTS #
    # ============================= #
    def test_DriftingGratingCircle_generate_movie_by_index(self):
        ref_rate = self.monitor.refresh_rate
        num_conditions = len(self.DGC._generate_all_conditions())
        block_frames = num_conditions*self.DGC.block_dur*ref_rate
        pregap_end = self.DGC.pregap_frame_num
        blocks_end = pregap_end + block_frames
        postgap_end = self.DGC.postgap_frame_num
        
        index_frames = pregap_end + block_frames + postgap_end
        
        assert (len(self.DGC_full_dict['stimulation']['index_to_display'])== index_frames)
    
    
if __name__ == '__main__':
    unittest.main(verbosity=2.)