import os
import unittest
import retinotopic_mapping.StimulusRoutines as sr

class TestSimulation(unittest.TestCase):

    def setUp(self):
        import retinotopic_mapping.MonitorSetup as ms

        # Setup monitor/indicator objects
        self.monitor = ms.Monitor(resolution=(1200,1600), dis=15.,
                                  mon_width_cm=40., mon_height_cm=30.,
                                  C2T_cm=15.,C2A_cm=20., center_coordinates=(0., 60.),
                                  downsample_rate=10)
        # import matplotlib.pyplot as plt
        # self.monitor.plot_map()
        # plt.show()

        self.indicator = ms.Indicator(self.monitor, width_cm = 3., height_cm = 3., position = 'northeast',
                                      is_sync = True, freq = 1.)

        self.curr_folder = os.path.dirname(os.path.realpath(__file__))

    def test_blur_cos(self):
        import numpy as np
        dis = np.arange(10, 30, 0.1) - 20.
        sigma = 10.
        blurred = sr.blur_cos(dis=dis, sigma=sigma)

        # import matplotlib.pyplot as plt
        # plt.plot(dis, blurred)
        # plt.show()

        # print blurred[50]
        # print blurred[100]

        assert (np.array_equal(blurred[0:50], np.ones((50,))))
        assert ((blurred[100] - 0.5) < 1E-10)
        assert (np.array_equal(blurred[150:200], np.zeros((50,))))

    def test_get_circle_mask(self):

        mask = sr.get_circle_mask(map_alt=self.monitor.deg_coord_y, map_azi=self.monitor.deg_coord_x,
                                  center=(10., 60.), radius=20., is_smooth_edge=True,
                                  blur_ratio=0.5, blur_func=sr.blur_cos, is_plot=False)
        # print mask[39, 100]
        assert (mask[39, 100] - 0.404847 < 1E10)

    def test_get_circle_mask2(self):
        import numpy as np

        alt = np.arange(-30., 30., 1.)[::-1]
        azi = np.arange(-30., 30., 1.)
        azi_map, alt_map = np.meshgrid(azi, alt)
        cm = sr.get_circle_mask(map_alt=alt_map, map_azi=azi_map, center=(0., 10.), radius=10.,
                                is_smooth_edge=False)
        # import matplotlib.pyplot as plt
        # plt.imshow(cm)
        # plt.show()
        assert (cm[28, 49] == 1)
        cm = sr.get_circle_mask(map_alt=alt_map, map_azi=azi_map, center=(10., 0.), radius=10.,
                                is_smooth_edge=False)
        # import matplotlib.pyplot as plt
        # plt.imshow(cm)
        # plt.show()
        assert (cm[10, 30] == 1)

    def test_get_warped_probes(self):

        import numpy as np
        azis = np.arange(0, 10, 0.1)
        alts = np.arange(30, 40, 0.1)[::-1]
        coord_azi, coord_alt = np.meshgrid(azis, alts)
        probes = ([32., 5., 1.],)

        frame = sr.get_warped_probes(deg_coord_alt=coord_alt, deg_coord_azi=coord_azi,
                                     probes=probes, width=0.5,
                                     height=1., ori=0., background_color=0.)

        # import matplotlib.pyplot as plt
        # plt.imshow(frame)
        # plt.show()
        assert (frame[75, 51] == 1)

        frame = sr.get_warped_probes(deg_coord_alt=coord_alt, deg_coord_azi=coord_azi,
                                     probes=probes, width=0.5,
                                     height=1., ori=30., background_color=0.)
        assert (frame[76, 47] == 1)
        assert (frame[81, 53] == 1)

    def test_get_grating(self):
        import numpy as np

        alt = np.arange(-30., 30., 1.)[::-1]
        azi = np.arange(-30., 30., 1.)
        azi_map, alt_map = np.meshgrid(azi, alt)

        grating = sr.get_grating(alt_map=alt_map, azi_map=azi_map, dire=315.,
                                 spatial_freq=0.04, center=(0., 0.), phase=0.,
                                 contrast=1.)
        assert (grating[34, 29] < 0.827)
        assert (grating[34, 29] > 0.825)

        # import matplotlib.pyplot as plt
        # f, (ax) = plt.subplots(1)
        # ax.imshow(grating, cmap='gray')
        # plt.show()

    def test_get_grid_locations(self):
        monitor_azi = self.monitor.deg_coord_x
        monitor_alt = self.monitor.deg_coord_y
        grid_locs = sr.get_grid_locations(subregion=[-20., -10., 30., 90.], grid_space=[10., 10.],
                                          monitor_azi=monitor_azi, monitor_alt=monitor_alt,
                                          is_include_edge=True, is_plot=False)
        assert (len(grid_locs) == 14)

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

    def test_FC_generate_movie_by_index(self):

        fc = sr.FlashingCircle(monitor=self.monitor,
                               indicator=self.indicator,
                               center=(10., 90.), flash_frame_num=30,
                               color=-1., pregap_dur=0.5, postgap_dur=1.2,
                               background=1., coordinate='degree',
                               midgap_dur=1., iteration=3)

        fc_full_seq, fc_full_dict = fc.generate_movie_by_index()

        assert (fc_full_seq.shape == (2, 120, 160))

        # print len(fc_full_dict['stimulation']['index_to_display'])
        assert (len(fc_full_dict['stimulation']['index_to_display']) == 312)

        frames_unique = fc_full_dict['stimulation']['frames_unique']
        frames = []
        for ind in fc_full_dict['stimulation']['index_to_display']:
            frames.append(frames_unique[ind])

        # Parameters defining where the frame blocks should start and end
        flashing_end = fc.pregap_frame_num + fc.flash_frame_num
        midgap_end = flashing_end + fc.midgap_frame_num
        next_flash_end = midgap_end + fc.flash_frame_num

        for i in range(fc.pregap_frame_num):
            assert (frames[i] == (0, -1.))

        for i in range(fc.pregap_frame_num, flashing_end):
            assert (frames[i] == (1, 1.))

        for i in range(flashing_end, midgap_end):
            assert (frames[i] == (0., -1.))

        for i in range(midgap_end, next_flash_end):
            assert (frames[i] == (1, 1.))

        assert (fc_full_seq[1, 39, 124] == -1)
        # import matplotlib.pyplot as plt
        # f, (ax) = plt.subplots(1)
        # ax.imshow(fc_full_seq[1])
        # plt.show()

    def test_FC_generate_movie(self):

        fc = sr.FlashingCircle(monitor=self.monitor,
                               indicator=self.indicator,
                               center=(10., 90.), flash_frame_num=30,
                               color=-1., pregap_dur=0.1, postgap_dur=1.0,
                               background=1., coordinate='degree',
                               midgap_dur=0.5, iteration=10)

        fc_full_seq, fc_full_dict = fc.generate_movie()

        assert (fc_full_seq.shape == (636, 120, 160))
        assert (len(fc_full_dict['stimulation']['frames']) == 636)

        frames = fc_full_dict['stimulation']['frames']
        # print frames

        # Parameters defining where the frame blocks should start and end
        flashing_end = fc.pregap_frame_num + fc.flash_frame_num
        midgap_end = flashing_end + fc.midgap_frame_num
        next_flash_end = midgap_end + fc.flash_frame_num

        for i in range(fc.pregap_frame_num):
            assert (frames[i] == (0, -1.))

        for i in range(fc.pregap_frame_num, flashing_end):
            assert (frames[i] == (1, 1.))

        for i in range(flashing_end, midgap_end):
            assert (frames[i] == (0., -1.))

        for i in range(midgap_end, next_flash_end):
            assert (frames[i] == (1, 1.))

        assert (fc_full_seq[6, 39, 124] == -1.)

        # import matplotlib.pyplot as plt
        # f, (ax) = plt.subplots(1)
        # ax.imshow(fc_full_seq[6])
        # plt.show()

    def test_SN_generate_display_index(self):
        sn = sr.SparseNoise(monitor=self.monitor, indicator=self.indicator,
                            background=0., coordinate='degree', grid_space=(10.,10.),
                            probe_size=(10.,10.), probe_orientation=0., probe_frame_num=6,
                            subregion=[10, 20, 0., 60.], sign='ON', iteration=1, pregap_dur=0.1,
                            postgap_dur=0.2, is_include_edge=True)

        frames_unique, index_to_display = sn._generate_display_index()
        for frame in frames_unique:
            assert (len(frame) == 4)
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

    def test_SN_get_probe_index_for_one_iter_on_off(self):
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
        for frame in frames_unique:
            assert (len(frame) == 4)
        assert (index_to_display[:30] == [0] * 30)
        assert (index_to_display[-18:] == [0] * 18)
        assert (max(index_to_display) == len(frames_unique) - 1)

        # frame_num_iter = (len(index_to_display) - 18 - 30) / 2
        assert ((len(index_to_display) - 48) % (8 * 2) == 0)
        probe_num = (len(index_to_display) - 48) / (8 * 2)
        for probe_ind in range(probe_num):
            assert (len(set(index_to_display[30 + probe_ind * 8: 34 + probe_ind * 8])) == 1)
            assert (len(set(index_to_display[34 + probe_ind * 8: 38 + probe_ind * 8])) == 1)
            assert (np.array_equal(frames_unique[index_to_display[33 + probe_ind * 8]][1],
                                   frames_unique[index_to_display[34 + probe_ind * 8]][1]))

    def test_SN_generate_movie_by_index(self):
        sn = sr.SparseNoise(monitor=self.monitor, indicator=self.indicator,
                            background=0., coordinate='degree', grid_space=(10., 10.),
                            probe_size=(10., 10.), probe_orientation=0., probe_frame_num=6,
                            subregion=[-20., -10., 30., 90.], sign='ON', iteration=1, pregap_dur=0.1,
                            postgap_dur=0.2, is_include_edge=True)
        mov_unique, _ = sn.generate_movie_by_index()
        import numpy as np
        # import matplotlib.pyplot as plt
        # plt.imshow(np.max(mov_unique, axis=0))
        # plt.show()
        assert (np.max(mov_unique, axis=0)[66, 121] == 1)

    def test_SN_generate_movie(self):
        sn = sr.SparseNoise(monitor=self.monitor, indicator=self.indicator,
                            background=0., coordinate='degree', grid_space=(10., 10.),
                            probe_size=(10., 10.), probe_orientation=0., probe_frame_num=6,
                            subregion=[-20., -10., 30., 90.], sign='OFF', iteration=1, pregap_dur=0.1,
                            postgap_dur=0.2, is_include_edge=True)
        mov, _ = sn.generate_movie()
        import numpy as np
        import matplotlib.pyplot as plt
        # plt.imshow(np.min(mov, axis=0))
        # plt.show()
        assert (np.min(mov, axis=0)[92, 38] == -1)

    def test_DGC_generate_frames(self):
        dgc = sr.DriftingGratingCircle(monitor=self.monitor, indicator=self.indicator, background=0.,
                                       coordinate='degree', center=(10., 90.), sf_list=(0.02, 0.04),
                                       tf_list=(1.0,), dire_list=(45.,), con_list=(0.8,), radius_list=(20.,),
                                       block_dur=2., midgap_dur=1., iteration=2, pregap_dur=1.5,
                                       postgap_dur=3., is_blank_block=False)

        frames = dgc.generate_frames()
        assert (len(frames) == 930)
        assert ([f[0] for f in frames[0:90]] == [0] * 90)
        assert ([f[0] for f in frames[210:270]] == [0] * 60)
        assert ([f[0] for f in frames[390:450]] == [0] * 60)
        assert ([f[0] for f in frames[570:630]] == [0] * 60)
        assert ([f[0] for f in frames[750:930]] == [0] * 180)
        assert ([f[8] for f in frames[0:90]] == [-1.] * 90)
        assert ([f[8] for f in frames[210:270]] == [-1.] * 60)
        assert ([f[8] for f in frames[390:450]] == [-1.] * 60)
        assert ([f[8] for f in frames[570:630]] == [-1.] * 60)
        assert ([f[8] for f in frames[750:930]] == [-1.] * 180)

        assert ([f[0] for f in frames[90:210]] == [1] * 120)
        assert ([f[0] for f in frames[270:390]] == [1] * 120)
        assert ([f[0] for f in frames[450:570]] == [1] * 120)
        assert ([f[0] for f in frames[630:750]] == [1] * 120)
        assert (frames[90][8] == 1.)
        assert ([f[8] for f in frames[91:150]] == [0.] * 59)
        assert (frames[150][8] == 1.)
        assert ([f[8] for f in frames[151:210]] == [0.] * 59)
        assert (frames[270][8] == 1.)
        assert ([f[8] for f in frames[271:330]] == [0.] * 59)
        assert (frames[330][8] == 1.)
        assert ([f[8] for f in frames[331:390]] == [0.] * 59)
        assert (frames[450][8] == 1.)
        assert ([f[8] for f in frames[451:510]] == [0.] * 59)
        assert (frames[510][8] == 1.)
        assert ([f[8] for f in frames[511:570]] == [0.] * 59)
        assert (frames[630][8] == 1.)
        assert ([f[8] for f in frames[631:690]] == [0.] * 59)
        assert (frames[690][8] == 1.)
        assert ([f[8] for f in frames[691:750]] == [0.] * 59)

    def test_DGC_blank_block(self):
        dgc = sr.DriftingGratingCircle(monitor=self.monitor, indicator=self.indicator, background=0.,
                                       coordinate='degree', center=(10., 90.), sf_list=(0.02,),
                                       tf_list=(4.0,), dire_list=(45.,), con_list=(0.8,), radius_list=(20.,),
                                       block_dur=0.5, midgap_dur=0.1, iteration=2, pregap_dur=0.2,
                                       postgap_dur=0.3, is_blank_block=True)

        frames = dgc.generate_frames()
        # print('\n'.join([str(f) for f in frames]))
        assert (len(frames) == 168)
        for frame in frames:
            assert (len(frame) == 9)

        _ = dgc._generate_frames_for_index_display_condition((0., 0., 0., 0., 0.))
        frames_unique_blank, index_to_display_blank = _
        # print('\nDGC frames_unique_blank:')
        # print('\n'.join([str(f) for f in frames_unique_blank]))
        # print('\nDGC index_to_display_blank:')
        # print(index_to_display_blank)

        assert (frames_unique_blank == ((1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                                        (1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        assert (index_to_display_blank == [0] + [1] * 29)

        frames_unique, condi_ind_in_frames_unique = dgc._generate_frames_unique_and_condi_ind_dict()
        # print('\nDGC frames_unique:')
        # print('\n'.join([str(f) for f in frames_unique]))
        # print('\nDGC condi_ind_in_frames_unique:')
        # print(condi_ind_in_frames_unique)
        assert (frames_unique[-1] == (1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert (frames_unique[-2] == (1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        assert (condi_ind_in_frames_unique['condi_0001'] == [16] + [17] * 29)

    def test_DGC_generate_frames_for_index_display_condition(self):
        dgc = sr.DriftingGratingCircle(monitor=self.monitor, indicator=self.indicator,
                                       block_dur=2., sf_list=(0.04,), tf_list=(2.0,),
                                       dire_list=(45.,), con_list=(0.8,), radius_list=(10.,),
                                       midgap_dur=0.1, pregap_dur=0.5, postgap_dur=0.2,
                                       iteration=2, is_blank_block=False)

        conditions = dgc._generate_all_conditions()
        # print len(conditions)
        assert (len(conditions) == 1)
        frames_unique_condi, index_to_display_condi = dgc._generate_frames_for_index_display_condition(conditions[0])
        assert (index_to_display_condi == range(30) * 4)
        assert (max(index_to_display_condi) == len(frames_unique_condi) - 1)
        # print '\n'.join([str(f) for f in frames_unique_condi])
        assert ([f[0] for f in frames_unique_condi] == [1] * 30)
        assert (frames_unique_condi[0][1] == 1)
        assert (frames_unique_condi[0][8] == 1.)
        assert ([f[1] for f in frames_unique_condi[1:]] == [0] * 29)
        assert ([f[8] for f in frames_unique_condi[1:]] == [0.] * 29)

    def test_DGC_generate_frames_unique_and_condi_ind_dict(self):
        dgc = sr.DriftingGratingCircle(monitor=self.monitor, indicator=self.indicator,
                                       block_dur=2., sf_list=(0.04,), tf_list=(1., 3.0,),
                                       dire_list=(45., 90.), con_list=(0.8,), radius_list=(10.,),
                                       midgap_dur=0.1, pregap_dur=0.5, postgap_dur=0.2,
                                       iteration=2, is_blank_block=False)
        frames_unique, condi_ind_in_frames_unique = dgc._generate_frames_unique_and_condi_ind_dict()
        assert (len(condi_ind_in_frames_unique) == 4)
        assert (set(condi_ind_in_frames_unique.keys()) == {'condi_0000', 'condi_0001', 'condi_0002', 'condi_0003'})
        assert (len(frames_unique) == 161)
        for frame in frames_unique:
            assert (len(frame) == 9)

        import numpy as np
        for cond, ind in condi_ind_in_frames_unique.items():
            assert (len(ind) == 120)
            assert (ind[0] % 20 == 1)
            assert (len(np.unique(ind)) == 60 or len(np.unique(ind)) == 20)
            # print '\ncond'
            # print ind

    def test_DGC_generate_display_index(self):
        dgc = sr.DriftingGratingCircle(monitor=self.monitor, indicator=self.indicator,
                                       block_dur=2., sf_list=(0.04,), tf_list=(1., 3.0,),
                                       dire_list=(45., 90.), con_list=(0.8,), radius_list=(10.,),
                                       midgap_dur=0.1, pregap_dur=0.5, postgap_dur=0.2,
                                       iteration=2, is_blank_block=False)
        frames_unique, index_to_display = dgc._generate_display_index()
        # print '\n'.join([str(f) for f in frames_unique])
        assert (len(frames_unique) == 161)
        assert (max(index_to_display) == len(frames_unique) - 1)
        # print len(index_to_display)
        assert (len(index_to_display) == 1044)

    def test_LSN_generate_all_probes(self):
        lsn = sr.LocallySparseNoise(monitor=self.monitor, indicator=self.indicator,
                                    min_distance=20., background=0., coordinate='degree',
                                    grid_space=(10., 10.), probe_size=(10., 10.),
                                    probe_orientation=0., probe_frame_num=6, subregion=[-10., 10., 0., 30.],
                                    sign='ON', iteration=1, pregap_dur=2., postgap_dur=3.,
                                    is_include_edge=True, repeat=1)

        all_probes = lsn._generate_all_probes()
        all_probes = [tuple(p) for p in all_probes]
        assert (set(all_probes) == {
                                    (-10., 0., 1.), (0., 0., 1.), (10., 0., 1.),
                                    (-10., 10., 1.), (0., 10., 1.), (10., 10., 1.),
                                    (-10., 20., 1.), (0., 20., 1.), (10., 20., 1.),
                                    (-10., 30., 1.), (0., 30., 1.), (10., 30., 1.),
                                    })

    def test_LSN_generate_probe_locs_one_frame(self):
        lsn = sr.LocallySparseNoise(monitor=self.monitor, indicator=self.indicator,
                                    min_distance=20., background=0., coordinate='degree',
                                    grid_space=(10.,10.), probe_size=(10.,10.),
                                    probe_orientation=0., probe_frame_num=6, subregion=[-10., 20., 0., 60.],
                                    sign='ON', iteration=1, pregap_dur=2., postgap_dur=3.,
                                    is_include_edge=True, repeat=1)

        all_probes = lsn._generate_all_probes()
        probes_one_frame, all_probes_left = lsn._generate_probe_locs_one_frame(all_probes)

        import itertools
        import numpy as np
        for (p0, p1) in itertools.combinations(probes_one_frame, r=2):
            curr_dis = np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) **2)
            # print (p0, p1), curr_dis
            assert (curr_dis > 20.)

    def test_LSN_generate_probe_sequence_one_iteration(self):
        lsn = sr.LocallySparseNoise(monitor=self.monitor, indicator=self.indicator,
                                    min_distance=20., background=0., coordinate='degree',
                                    grid_space=(10., 10.), probe_size=(10., 10.),
                                    probe_orientation=0., probe_frame_num=6, subregion=[-10., 20., 0., 60.],
                                    sign='ON-OFF', iteration=1, pregap_dur=2., postgap_dur=3.,
                                    is_include_edge=True, repeat=1)

        all_probes = lsn._generate_all_probes()
        frames = lsn._generate_probe_sequence_one_iteration(all_probes=all_probes, is_redistribute=False)
        # print '\n'.join([str(f) for f in frames])
        # print [len(f) for f in frames]
        assert (sum([len(f) for f in frames]) == len(all_probes))

        import itertools
        import numpy as np
        alt_lst = np.arange(-10., 25., 10)
        azi_lst = np.arange(0., 65., 10)
        all_probes = list(itertools.product(alt_lst, azi_lst, [-1., 1.]))
        all_probes_frame = []

        for frame in frames:
            all_probes_frame += [tuple(probe) for probe in frame]
            # asserting all pairs in the particular frame meet sparsity criterion
            for (p0, p1) in itertools.combinations(frame, r=2):
                curr_dis = np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
                # print (p0, p1), curr_dis
                assert (curr_dis > 20.)

        # assert all frames combined cover whole subregion
        assert (set(all_probes) == set(all_probes_frame))

    def test_LSN_is_fit(self):
        # todo: finish this
        pass

    def test_LSN_redistribute_one_probe(self):
        # todo: finish this
        pass

    def test_LSN_redistribute_probes(self):
        lsn = sr.LocallySparseNoise(monitor=self.monitor, indicator=self.indicator,
                                    min_distance=20., background=0., coordinate='degree',
                                    grid_space=(10., 10.), probe_size=(10., 10.),
                                    probe_orientation=0., probe_frame_num=6, subregion=[-10., 20., 0., 60.],
                                    sign='ON-OFF', iteration=1, pregap_dur=2., postgap_dur=3.,
                                    is_include_edge=True, repeat=1)

        all_probes = lsn._generate_all_probes()
        frames = lsn._generate_probe_sequence_one_iteration(all_probes=all_probes, is_redistribute=True)
        # print '\n'.join([str(f) for f in frames])
        # print [len(f) for f in frames]
        assert (sum([len(f) for f in frames]) == len(all_probes))

        import itertools
        import numpy as np
        alt_lst = np.arange(-10., 25., 10)
        azi_lst = np.arange(0., 65., 10)
        all_probes = list(itertools.product(alt_lst, azi_lst, [-1., 1.]))
        all_probes_frame = []

        for frame in frames:
            all_probes_frame += [tuple(probe) for probe in frame]
            # asserting all pairs in the particular frame meet sparsity criterion
            for (p0, p1) in itertools.combinations(frame, r=2):
                curr_dis = np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
                # print (p0, p1), curr_dis
                assert (curr_dis > 20.)

        # assert all frames combined cover whole subregion
        assert (set(all_probes) == set(all_probes_frame))

    def test_LSN_generate_frames_for_index_display(self):
        lsn = sr.LocallySparseNoise(monitor=self.monitor, indicator=self.indicator,
                                    min_distance=20., background=0., coordinate='degree',
                                    grid_space=(10., 10.), probe_size=(10., 10.),
                                    probe_orientation=0., probe_frame_num=6, subregion=[-10., 20., 0., 60.],
                                    sign='ON-OFF', iteration=2, pregap_dur=2., postgap_dur=3.,
                                    is_include_edge=True, repeat=1)

        frames_unique = lsn._generate_frames_for_index_display()
        # print len(frames_unique)
        # print '\n'.join([str(f) for f in frames_unique])
        assert (len(frames_unique) % 2 == 1)
        for frame in frames_unique:
            assert (len(frame) == 4)

    def test_LSN_generate_display_index(self):
        lsn = sr.LocallySparseNoise(monitor=self.monitor, indicator=self.indicator,
                                    min_distance=20., background=0., coordinate='degree',
                                    grid_space=(10., 10.), probe_size=(10., 10.),
                                    probe_orientation=30., probe_frame_num=6, subregion=[-10., 20., 0., 60.],
                                    sign='ON-OFF', iteration=2, pregap_dur=2., postgap_dur=3.,
                                    is_include_edge=True, repeat=1)
        frames_unique, index_to_display = lsn._generate_display_index()
        # print index_to_display
        assert (index_to_display[:lsn.pregap_frame_num] == [0] * lsn.pregap_frame_num)
        assert (index_to_display[-lsn.postgap_frame_num:] == [0] * lsn.postgap_frame_num)
        assert (len(index_to_display) == (len(frames_unique) - 1) * lsn.probe_frame_num / 2 +
                lsn.pregap_frame_num + lsn.postgap_frame_num)

    def test_LSN_repeat(self):
        lsn = sr.LocallySparseNoise(monitor=self.monitor, indicator=self.indicator,
                                    min_distance=20., background=0., coordinate='degree',
                                    grid_space=(10., 10.), probe_size=(10., 10.),
                                    probe_orientation=0., probe_frame_num=4, subregion=[-10., 20., 0., 60.],
                                    sign='ON-OFF', iteration=1, pregap_dur=2., postgap_dur=3.,
                                    is_include_edge=True, repeat=3)

        import itertools
        import numpy as np
        alt_lst = np.arange(-10., 25., 10)
        azi_lst = np.arange(0., 65., 10)
        all_probes = list(itertools.product(alt_lst, azi_lst, [-1., 1.]))

        frames_unique, display_index = lsn._generate_display_index()
        for probe in all_probes:
            present_frames = 0
            for di in display_index:
                if frames_unique[di][1] is not None and list(probe) in frames_unique[di][1]:
                    present_frames += 1
            # print('probe:{}, number of frames: {}'.format(str(probe), present_frames))
            assert (present_frames == 4 * 3)

    def test_SGC_generate_frames_for_index_display(self):
        sgc = sr.StaticGratingCircle(monitor=self.monitor, indicator=self.indicator, background=0.,
                                     coordinate='degree', center=(0., 30.), sf_list=(0.02, 0.04, 0.08),
                                     ori_list=(0., 45., 90., 135.), con_list=(0.2, 0.5, 0.8),
                                     radius_list=(50.,), phase_list=(0., 90., 180., 270.),
                                     display_dur=0.25, midgap_dur=0., iteration=2, pregap_dur=2.,
                                     postgap_dur=3., is_blank_block=False)
        frames_unique = sgc._generate_frames_for_index_display()
        # print len(frames_unique)
        assert (len(frames_unique) == (3 * 4 * 3 * 4 * 2 + 1))
        for frame in frames_unique:
            assert(len(frame) == 7)

        sgc = sr.StaticGratingCircle(monitor=self.monitor, indicator=self.indicator, background=0.,
                                     coordinate='degree', center=(0., 30.), sf_list=(0.02, 0.04, 0.08),
                                     ori_list=(0., 90., 180., 270.), con_list=(0.2, 0.5, 0.8),
                                     radius_list=(50.,), phase_list=(0., 90., 180., 270.),
                                     display_dur=0.25, midgap_dur=0., iteration=2, pregap_dur=2.,
                                     postgap_dur=3., is_blank_block=False)
        frames_unique = sgc._generate_frames_for_index_display()
        # print len(frames_unique)
        assert (len(frames_unique) == (3 * 2 * 3 * 4 * 2 + 1))

    def test_SGC_generate_display_index(self):
        sgc = sr.StaticGratingCircle(monitor=self.monitor, indicator=self.indicator, background=0.,
                                     coordinate='degree', center=(0., 30.), sf_list=(0.02, 0.04, 0.08),
                                     ori_list=(0., 45., 90., 135.), con_list=(0.2, 0.5, 0.8),
                                     radius_list=(50.,), phase_list=(0., 90., 180., 270.),
                                     display_dur=0.25, midgap_dur=0.1, iteration=2, pregap_dur=2.,
                                     postgap_dur=3., is_blank_block=False)
        frames_unique, index_to_display = sgc._generate_display_index()
        for frame in frames_unique:
            assert (len(frame) == 7)
        assert (max(index_to_display) == len(frames_unique) - 1)
        # print len(index_to_display)
        # print index_to_display
        assert (len(index_to_display) == 6342)

    def test_SGC_blank_block(self):
        sgc = sr.StaticGratingCircle(monitor=self.monitor, indicator=self.indicator, background=0.,
                                     coordinate='degree', center=(0., 30.), sf_list=(0.04,),
                                     ori_list=(90., ), con_list=(0.8, ), radius_list=(50.,),
                                     phase_list=(0., 180.,), display_dur=0.1, midgap_dur=0.1,
                                     iteration=2, pregap_dur=0., postgap_dur=0., is_blank_block=True)
        all_conditions = sgc._generate_all_conditions()
        # print('\nSGC all_conditions:')
        # print('\n'.join([str(c) for c in all_conditions]))
        assert (all_conditions[-1] == (0., 0., 0., 0., 0.))

        frames_unique = sgc._generate_frames_for_index_display()
        for frame in frames_unique:
            assert (len(frame) == 7)
        # print('\nSGC frames_unique:')
        # print('\n'.join([str(f) for f in frames_unique]))
        assert (frames_unique[-1] == (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert (frames_unique[-2] == (1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        _, index_to_display = sgc._generate_display_index()
        assert (len(index_to_display) == 66)

    def test_SS_generate_display_index(self):
        ss = sr.StimulusSeparator(monitor=self.monitor, indicator=self.indicator,
                                  coordinate='degree', background=0.,
                                  indicator_on_frame_num=4, indicator_off_frame_num=4,
                                  cycle_num=10, pregap_dur=0., postgap_dur=0.)

        frames_unique, index_to_display = ss._generate_display_index()
        assert (frames_unique == ((0, -1), (1, 1.), (1, -1.)))
        assert (len(index_to_display) == 80)
        for frame in frames_unique:
            assert (len(frame) == 2)

    def test_SI_wrap_images(self):
        si = sr.StaticImages(monitor=self.monitor, indicator=self.indicator, background=0.,
                             coordinate='degree', img_center=(0., 60.), deg_per_pixel=(0.1, 0.1),
                             display_dur=0.25, midgap_dur=0., iteration=1, pregap_dur=2.,
                             postgap_dur=3., is_blank_block=False)

        img_w_path = os.path.join(self.curr_folder, 'test_data', 'wrapped_images_for_display.hdf5')

        if os.path.isfile(img_w_path):
            os.remove(img_w_path)

        si.wrap_images(work_dir=os.path.join(self.curr_folder, 'test_data'))

        import h5py
        img_w_f = h5py.File(img_w_path, 'r')

        assert (img_w_f['images_wrapped/images'].shape == (2, 120, 160))
        assert (img_w_f['images_wrapped/altitude'].shape == (120, 160))
        assert (img_w_f['images_wrapped/azimuth'].shape == (120, 160))
        import numpy as np
        assert (np.array_equal(img_w_f['images_wrapped/altitude'].value, self.monitor.deg_coord_y))
        assert (np.array_equal(img_w_f['images_wrapped/azimuth'].value, self.monitor.deg_coord_x))

        assert (img_w_f['images_dewrapped/images'].shape == (2, 270, 473))
        assert (img_w_f['images_dewrapped/altitude'].shape == (270, 473))
        assert (img_w_f['images_dewrapped/azimuth'].shape == (270, 473))

        img_w_f.close()

        os.remove(img_w_path)

    def test_SI_generate_frames_for_index_display(self):
        si = sr.StaticImages(monitor=self.monitor, indicator=self.indicator, background=0.,
                             coordinate='degree', img_center=(0., 60.), deg_per_pixel=(0.1, 0.1),
                             display_dur=0.25, midgap_dur=0., iteration=1, pregap_dur=2.,
                             postgap_dur=3., is_blank_block=False)
        import numpy as np
        si.images_wrapped = np.random.rand(27, 120, 160)
        frames_unique = si._generate_frames_for_index_display()
        assert (len(frames_unique) == 55)
        for frame in frames_unique:
            assert (len(frame) == 3)

    def test_SI_generate_display_index(self):
        si = sr.StaticImages(monitor=self.monitor, indicator=self.indicator, background=0.,
                             coordinate='degree', img_center=(0., 60.), deg_per_pixel=(0.1, 0.1),
                             display_dur=0.25, midgap_dur=0.1, iteration=2, pregap_dur=2.,
                             postgap_dur=3., is_blank_block=False)
        import numpy as np
        si.images_wrapped = np.random.rand(15, 120, 160)
        frames_unique, index_to_display = si._generate_display_index()
        assert (len(index_to_display) == 924)
        for frame in frames_unique:
            assert (len(frame) == 3)

    def test_SI_blank_block(self):
        si = sr.StaticImages(monitor=self.monitor, indicator=self.indicator, background=0.,
                             coordinate='degree', img_center=(0., 60.), deg_per_pixel=(0.1, 0.1),
                             display_dur=0.1, midgap_dur=0.1, iteration=1, pregap_dur=0.,
                             postgap_dur=0., is_blank_block=True)
        import numpy as np
        si.images_wrapped = np.random.rand(2, 120, 160)
        frames_unique, index_to_display = si._generate_display_index()
        assert (len(frames_unique) == 7)
        for frame in frames_unique:
            assert (len(frame) == 3)
        assert (frames_unique[-1] == (1, -1, 0.))
        assert (frames_unique[-2] == (1, -1, 1.))

        # print('frames_unique:')
        # print('\n'.join([str(f) for f in frames_unique]))
        # print('\nindex_to_display: {}.'.format(index_to_display))
        # print('\nframes to be displayed:')
        # frames = [frames_unique[i] for i in index_to_display]
        # print('\n'.join([str(f) for f in frames]))
        assert (len(index_to_display) == 30)


if __name__ == '__main__':
    unittest.main(verbosity=2.)