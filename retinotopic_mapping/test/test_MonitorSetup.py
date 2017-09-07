import os
import unittest
import retinotopic_mapping.MonitorSetup as ms

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestSimulation(unittest.TestCase):

    def setUp(self):
        import tifffile as tf
        self.natural_scene = tf.imread(os.path.join(curr_folder,
                                                    'test_data',
                                                    'natural_scene.tif'))

    def test_Monitor_generate_lookup_table(self):
        mon = ms.Monitor(resolution=(1200,1600), dis=15.,
                         mon_width_cm=40., mon_height_cm=30.,
                         C2T_cm=15.,C2A_cm=20., mon_tilt=30.,
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
                         C2T_cm=15., C2A_cm=20., mon_tilt=30.,
                         downsample_rate=10)
        import numpy as np
        nsw = mon.warp_images(imgs=np.array([self.natural_scene]),
                              center_coor=[0., 60.], deg_per_pixel=0.2,
                              is_luminance_correction=True)

        # import matplotlib.pyplot as plt
        # plt.imshow(nsw[0], cmap='gray', vmin=0, vmax=255)
        # plt.show()
