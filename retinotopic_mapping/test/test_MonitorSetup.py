import os
import unittest
import retinotopic_mapping.MonitorSetup as ms

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestMonitorSetup(unittest.TestCase):

    def setUp(self):
        import skimage.external.tifffile as tf
        self.natural_scene = tf.imread(os.path.join(curr_folder,
                                                    'test_data',
                                                    'images_original.tif'))

    def test_Monitor_remap(self):
        mon = ms.Monitor(resolution=(1200, 1600), dis=15.,
                         mon_width_cm=40., mon_height_cm=30.,
                         C2T_cm=15., C2A_cm=20., center_coordinates=(0., 60.),
                         downsample_rate=10)
        mon.remap()
        assert(abs(mon.deg_coord_y[60, 80] - 0.) < 1.)
        assert(abs(mon.deg_coord_x[60, 80] - 60.) < 1.)

        # mon.plot_map()
        # import matplotlib.pyplot as plt
        # plt.show()

        mon = ms.Monitor(resolution=(1200, 1600), dis=15.,
                         mon_width_cm=40., mon_height_cm=30.,
                         C2T_cm=15., C2A_cm=20., center_coordinates=(20., -10.),
                         downsample_rate=10)
        mon.remap()
        assert (abs(mon.deg_coord_y[60, 80] - 20.) < 1.)
        assert (abs(mon.deg_coord_x[60, 80] - (-10.)) < 1.)
        # mon.plot_map()
        # import matplotlib.pyplot as plt
        # plt.show()

        mon = ms.Monitor(resolution=(1200, 1600), dis=15.,
                         mon_width_cm=40., mon_height_cm=30.,
                         C2T_cm=5., C2A_cm=35., center_coordinates=(20., -10.),
                         downsample_rate=10)
        mon.remap()
        assert (abs(mon.deg_coord_y[20, 140] - 20.) < 1.)
        assert (abs(mon.deg_coord_x[20, 140] - (-10.)) < 1.)
        # mon.plot_map()
        # import matplotlib.pyplot as plt
        # plt.show()

    def test_Monitor_generate_lookup_table(self):
        mon = ms.Monitor(resolution=(1200,1600), dis=15.,
                         mon_width_cm=40., mon_height_cm=30.,
                         C2T_cm=15.,C2A_cm=20., center_coordinates=(0., 60.),
                         downsample_rate=10)

        lookup_i, lookup_j = mon.generate_lookup_table()

        # import matplotlib.pyplot as plt
        # f, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        # fig0 = ax0.imshow(lookup_i)
        # f.colorbar(fig0, ax=ax0)
        # fig1 = ax1.imshow(lookup_j)
        # f.colorbar(fig1, ax=ax1)
        # plt.show()

    def test_Monitor_warp_images(self):
        mon = ms.Monitor(resolution=(1200, 1600), dis=15.,
                         mon_width_cm=40., mon_height_cm=30.,
                         C2T_cm=15., C2A_cm=20., center_coordinates=(0., 60.),
                         downsample_rate=10)
        import numpy as np
        nsw, altw, aziw, nsd, altd, azid = mon.warp_images(imgs=np.array([self.natural_scene]),
                                                           center_coor=[0., 60.], deg_per_pixel=0.2,
                                                           is_luminance_correction=True)

        # import matplotlib.pyplot as plt
        # f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
        # fig1 = ax1.imshow(self.natural_scene, cmap='gray', vmin=0., vmax=255.)
        # ax1.set_axis_off()
        # ax1.set_title('original')
        # f.colorbar(fig1, ax=ax1)
        # fig2 = ax2.imshow(nsw[0], cmap='gray', vmin=-1., vmax=1.)
        # ax2.set_axis_off()
        # ax2.set_title('wrapped')
        # f.colorbar(fig2, ax=ax2)
        # fig3 = ax3.imshow(nsd[0], cmap='gray', vmin=0, vmax=255)
        # ax3.set_axis_off()
        # ax3.set_title('dewrapped')
        # f.colorbar(fig3, ax=ax3)
        # plt.show()
        #
        # print altd.shape
        # print azid.shape

        assert (altw.shape[0] == nsw.shape[1])
        assert (altw.shape[1] == nsw.shape[2])
        assert (aziw.shape[0] == nsw.shape[1])
        assert (aziw.shape[1] == nsw.shape[2])
        assert (altd.shape[0] == nsd.shape[1])
        assert (altd.shape[1] == nsd.shape[2])
        assert (azid.shape[0] == nsd.shape[1])
        assert (azid.shape[1] == nsd.shape[2])
        assert (np.nanmean(nsw.flat) < 1E6)
