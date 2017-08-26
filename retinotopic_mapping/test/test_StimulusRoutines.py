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

    def test_get_grating(self):
        import numpy as np
        import matplotlib.pyplot as plt
        alt = np.arange(-30., 30., 1.)
        azi = np.arange(-30., 30., 1.)
        azi_map, alt_map = np.meshgrid(azi, alt)

        grating = sr.get_grating(alt_map, azi_map, ori=45., spatial_freq=0.04,
                                 center=(0., 0.), phase=np.pi/2, contrast=1.)
        # f = plt.figure()
        # ax = f.add_subplot(111)
        # ax.imshow(grating, cmap='gray')
        # ax.set_xticks(range(len(azi))[::10])
        # ax.set_yticks(range(len(alt))[::10])
        # ax.set_xticklabels(azi[::10])
        # ax.set_yticklabels(alt[::10])
        # plt.show()

    # UNIFORM CONTRAST TESTS
    # ======================
    def test_UC_generate_movie_by_index(self):
        # Setup Uniform Contrast Objects
        uc = sr.UniformContrast(monitor=self.monitor, indicator=self.indicator, duration=0.1,
                                color=1., pregap_dur=1., postgap_dur=1.5, background=0.,
                                coordinate='degree')

        uc_full_seq, uc_full_dict = uc.generate_movie_by_index()

        assert (uc_full_seq.shape == (2, 120, 160))
        assert (len(uc_full_dict['stimulation']['index_to_display']) == 156)

        frames_unique = uc_full_dict['stimulation']['frames_unique']
        all_frames = []
        for ind in uc_full_dict['stimulation']['index_to_display']:
            all_frames.append(frames_unique[ind])

        # Parameters defining where the frame blocks should start and end
        ref_rate = self.monitor.refresh_rate
        pregap_end = uc.pregap_frame_num
        on_end = pregap_end + int(uc.duration*ref_rate)
        postgap_end = on_end + uc.postgap_frame_num
        
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
        fc = sr.FlashingCircle(monitor=self.monitor,
                                    indicator=self.indicator,
                                    center=(90., 0.), flash_frame_num=30,
                                    color=-1., pregap_dur=0.5, postgap_dur=1.2,
                                    background=1., coordinate='degree')

        fc_full_seq, fc_full_dict = fc.generate_movie_by_index()

        assert (fc_full_seq.shape == (2, 120, 160))

        assert (len(fc_full_dict['stimulation']['index_to_display']) == 132)

        frames_unique = fc_full_dict['stimulation']['frames_unique']
        frames = []
        for ind in fc_full_dict['stimulation']['index_to_display']:
            frames.append(frames_unique[ind])
        
        # Parameters defining where the frame blocks should start and end
        pregap_end = fc.pregap_frame_num
        flash_frames= fc.flash_frame_num
        flashing_end = pregap_end + flash_frames
        postgap_end = flashing_end + fc.postgap_frame_num

        for i in range(pregap_end):
            assert (frames[i] == (0., -1.))

        for i in range(pregap_end, flashing_end):
            assert (frames[i] == (1., 1.))

        for i in range(flashing_end, postgap_end):
            assert (frames[i] == (0., -1.))

    def test_FC_generate_movie(self):
        fc = sr.FlashingCircle(monitor=self.monitor,
                                    indicator=self.indicator,
                                    center=(90., 0.), flash_frame_num=30,
                                    color=-1., pregap_dur=0.1, postgap_dur=1.0,
                                    background=1., coordinate='degree')

        fc_full_seq, fc_full_dict = fc.generate_movie()

        assert (fc_full_seq.shape == (96, 120, 160))
        assert (len(fc_full_dict['stimulation']['frames']) == 96)

        frames = fc_full_dict['stimulation']['frames']
        # print frames

        # Parameters defining where the frame blocks should start and end
        pregap_end = fc.pregap_frame_num
        flash_frames = fc.flash_frame_num
        flashing_end = pregap_end + flash_frames
        postgap_end = flashing_end + fc.postgap_frame_num

        for i in range(pregap_end):
            assert (frames[i] == (0., -1.))

        for i in range(pregap_end, flashing_end):
            assert (frames[i] == (1., 1.))

        for i in range(flashing_end, postgap_end):
            assert (frames[i] == (0., -1.))
    
    # SPARSE NOISE TESTS #
    # ================== #
    def test_SN_generate_display_index(self):
        sn = sr.SparseNoise(monitor=self.monitor, indicator=self.indicator,
                            background=0., coordinate='degree', grid_space=(10.,10.),
                            probe_size=(10.,10.), probe_orientation=0., probe_frame_num=6,
                            subregion=[-10, 10, 45., 55.], sign='ON', iteration=1, pregap_dur=0.1,
                            postgap_dur=0.2, is_include_edge=True)

        frames_unique, index_to_display = sn._generate_display_index()
        # print '\n'.join([str(f) for f in frames_unique])
        # print index_to_display
        assert (index_to_display[:6] == [0, 0, 0, 0, 0, 0])
        assert (index_to_display[-12:] == [0] * 12)
        # print max(index_to_display)
        # print len(frames_unique)
        assert (max(index_to_display) == len(frames_unique) -1)
        probe_num = (len(index_to_display) - 18) / 6
        for probe_ind in range(probe_num):
            assert (len(set(index_to_display[6 + probe_ind * 6: 9 + probe_ind * 6])) == 1)
            assert (len(set(index_to_display[9 + probe_ind * 6: 12 + probe_ind * 6])) == 1)
            assert (index_to_display[9 + probe_ind * 6] - index_to_display[8 + probe_ind * 6] == 1)

    def test_SN__get_probe_index_for_one_iter_on_off(self):
        import numpy as np
        sn = sr.SparseNoise(monitor=self.monitor, indicator=self.indicator,
                            background=0., coordinate='degree', grid_space=(5., 5.),
                            probe_size=(5., 5.), probe_orientation=0., probe_frame_num=6,
                            subregion=[-30, 30, -10., 90.], sign='ON-OFF', iteration=2)
        frames_unique = sn._generate_frames_for_index_display()
        probe_ind = sn._get_probe_index_for_one_iter_on_off(frames_unique)
        for j in range(len(probe_ind) - 1):
            probe_loc_0 = frames_unique[probe_ind[j]]
            probe_loc_1 = frames_unique[probe_ind[j + 1]]
            assert(not np.array_equal(probe_loc_0, probe_loc_1))

    def test_SN_generate_display_index2(self):
        import numpy as np
        sn = sr.SparseNoise(monitor=self.monitor, indicator=self.indicator,
                            background=0., coordinate='degree', grid_space=(10., 10.),
                            probe_size=(10., 10.), probe_orientation=0., probe_frame_num=8,
                            subregion=[-10, 10, 45., 55.], sign='ON-OFF', iteration=2,
                            pregap_dur=0.5, postgap_dur=0.3, is_include_edge=True)

        frames_unique, index_to_display = sn._generate_display_index()
        assert (index_to_display[:30] == [0] * 30)
        assert (index_to_display[-18:] == [0] * 18)
        assert (max(index_to_display) == len(frames_unique) - 1)

        frame_num_iter = len(index_to_display) / 2
        assert (index_to_display[frame_num_iter - 18: frame_num_iter + 30] == [0] * 48)
        probe_num = (len(index_to_display[:frame_num_iter]) - 30) / 6
        for probe_ind in range(probe_num):
            assert (len(set(index_to_display[30 + probe_ind * 8: 34 + probe_ind * 8])) == 1)
            assert (len(set(index_to_display[34 + probe_ind * 8: 38 + probe_ind * 8])) == 1)
            assert (np.array_equal(frames_unique[index_to_display[33 + probe_ind * 8]][1],
                                   frames_unique[index_to_display[34 + probe_ind * 8]][1]))

    # DRIFTING GRATING CIRCLE TESTS #
    # ============================= #
    def test_DGC_generate_movie_by_index(self):
        dgc = sr.DriftingGratingCircle(monitor=self.monitor, indicator=self.indicator,
                                       block_dur=2., sf_list=(0.04,), tf_list=(2.0,),
                                       dire_list=(45.,), con_list=(0.8,), size_list=(10.,),
                                       midgap_dur=0.1, pregap_dur=0.5, postgap_dur=0.2,
                                       iteration=2)

        frames_unique, index_to_display = dgc._generate_display_index()
        print len(index_to_display)
        assert (len(index_to_display) == 2 * (30 + 120 + 12))

    
if __name__ == '__main__':
    unittest.main(verbosity=2.)