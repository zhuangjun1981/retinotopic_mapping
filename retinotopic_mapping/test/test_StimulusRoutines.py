import os
import unittest
import retinotopic_mapping.StimulusRoutines as sr

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestSimulation(unittest.TestCase):

    def setUp(self):
        import retinotopic_mapping.MonitorSetup as ms
        
        # Setup monitor/indicator objects
        self.monitor = ms.Monitor(resolution=(1200,1600), dis=15., 
                                   mon_width_cm=40., mon_height_cm=30., 
                                   C2T_cm=15.,C2A_cm=20., mon_tilt=30., downsample_rate=10)
        self.indicator = ms.Indicator(self.monitor, width_cm = 3., height_cm = 3., position = 'northeast',
                                      is_sync = True, freq = 1.)

    # UNIFORM CONTRAST TESTS
    # ======================
    def test_UC_generate_movie_by_index(self):
        # Setup Uniform Contrast Objects
        self.UC = sr.UniformContrast(monitor=self.monitor,
                                     indicator=self.indicator,
                                     duration=0.1, color=1.,
                                     pregap_dur=1., postgap_dur=1.5,
                                     background=0., coordinate='degree')

        self.UC_full_seq, self.UC_full_dict = self.UC.generate_movie_by_index()

        assert (self.UC_full_seq.shape == (2, 120, 160))
        assert (len(self.UC_full_dict['stimulation']['index_to_display']) == 156)

        frames_unique = self.UC_full_dict['stimulation']['frames_unique']
        all_frames = []
        for ind in self.UC_full_dict['stimulation']['index_to_display']:
            all_frames.append(frames_unique[ind])

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
    def test_FC_generate_movie_by_index(self):
        # Setup Flashing Circle Objects
        self.FC = sr.FlashingCircle(monitor=self.monitor,
                                    indicator=self.indicator,
                                    center=(90., 0.), flash_frame_num=30,
                                    color=-1., pregap_dur=0.5, postgap_dur=1.2,
                                    background=1., coordinate='degree')

        self.FC_full_seq, self.FC_full_dict = self.FC.generate_movie_by_index()

        assert (self.FC_full_seq.shape == (2, 120, 160))

        assert (len(self.FC_full_dict['stimulation']['index_to_display']) == 132)

        frames_unique = self.FC_full_dict['stimulation']['frames_unique']
        frames = []
        for ind in self.FC_full_dict['stimulation']['index_to_display']:
            frames.append(frames_unique[ind])
        
        # Parameters defining where the frame blocks should start and end
        pregap_end = self.FC.pregap_frame_num
        flash_frames= self.FC.flash_frame_num
        flashing_end = pregap_end + flash_frames
        postgap_end = flashing_end + self.FC.postgap_frame_num

        for i in range(pregap_end):
            assert (frames[i] == (0., -1.))

        for i in range(pregap_end, flashing_end):
            assert (frames[i] == (1., 1.))

        for i in range(flashing_end, postgap_end):
            assert (frames[i] == (0., -1.))

    def test_FC_generate_movie(self):
        self.FC = sr.FlashingCircle(monitor=self.monitor,
                                    indicator=self.indicator,
                                    center=(90., 0.), flash_frame_num=30,
                                    color=-1., pregap_dur=0.1, postgap_dur=1.0,
                                    background=1., coordinate='degree')

        self.FC_full_seq, self.FC_full_dict = self.FC.generate_movie()

        assert (self.FC_full_seq.shape == (96, 120, 160))
        assert (len(self.FC_full_dict['stimulation']['frames']) == 96)

        frames = self.FC_full_dict['stimulation']['frames']
        # print frames

        # Parameters defining where the frame blocks should start and end
        pregap_end = self.FC.pregap_frame_num
        flash_frames = self.FC.flash_frame_num
        flashing_end = pregap_end + flash_frames
        postgap_end = flashing_end + self.FC.postgap_frame_num

        for i in range(pregap_end):
            assert (frames[i] == (0., -1.))

        for i in range(pregap_end, flashing_end):
            assert (frames[i] == (1., 1.))

        for i in range(flashing_end, postgap_end):
            assert (frames[i] == (0., -1.))
    
    # SPARSE NOISE TESTS #
    # ================== #
    def test_SN_generate_movie_by_index(self):

        # Setup Sparse Noise Objects
        self.SN = sr.SparseNoise(monitor=self.monitor,
                                 indicator=self.indicator,
                                 probe_frame_num=10)
        self.SN_full_seq, self.SN_full_dict = self.SN.generate_movie_by_index()

        stim_on_frames = len(self.SN._generate_grid_points_sequence())*self.SN.probe_frame_num
        index_frames = self.SN.pregap_frame_num + stim_on_frames + self.SN.postgap_frame_num
        
        assert (len(self.SN_full_dict['stimulation']['index_to_display']) == index_frames)

    # DRIFTING GRATING CIRCLE TESTS #
    # ============================= #
    def test_DGC_generate_movie_by_index(self):

        # Setup Drifting Grating Circle objects
        self.DGC = sr.DriftingGratingCircle(monitor=self.monitor,
                                            indicator=self.indicator,
                                            sf_list=(0.08,),
                                            tf_list=(4.0,),
                                            dire_list=(0.,),
                                            con_list=(1.,),
                                            size_list=(10.,))
        self.DGC_full_seq, self.DGC_full_dict = self.DGC.generate_movie_by_index()

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