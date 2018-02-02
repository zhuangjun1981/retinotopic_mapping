import os
import unittest
import numpy as np
import retinotopic_mapping.tools.GenericTools as gt

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestMonitorSetup(unittest.TestCase):

    def setUp(self):
        self.data = np.array([1.0, 0.9, 0.8, 0.7, 0.9, 1.2, 1.5, 0.5, 0.6, 0.6, 0.6, 0.9, 0.9, 0.9, 1.2,
                              -0.3, -0.3, -0.2, 0.3, 2.0, 3.5, 0.8, 0.8, 0.6, 3.2, 1.4, 0.9, 0.9, 0.4])

    def test_up_crossings(self):
        assert (np.array_equal(gt.up_crossings(data=self.data, threshold=0.9), [5, 14, 19, 24]))
        assert (np.array_equal(gt.up_crossings(data=self.data, threshold=0.6), [11, 19, 24]))
        assert (np.array_equal(gt.up_crossings(data=self.data, threshold=0.5), [8, 19]))
        assert (len(gt.up_crossings(data=self.data, threshold=5.)) == 0)

    def test_down_crossings(self):
        assert (np.array_equal(gt.down_crossings(data=self.data, threshold=0.6), [7, 15, 28]))
        assert (np.array_equal(gt.down_crossings(data=self.data, threshold=0.5), [15, 28]))
        assert (np.array_equal(gt.down_crossings(data=self.data, threshold=2.), [21, 25]))
        assert (np.array_equal(gt.down_crossings(data=self.data, threshold=0.8), [3, 7, 15, 23, 28]))
        assert (len(gt.down_crossings(data=self.data, threshold=-3.)) == 0)

    def test_all_crossings(self):
        assert (np.array_equal(gt.all_crossings(data=self.data, threshold=0.6), [7, 11, 15, 19, 24, 28]))
        assert (len(gt.all_crossings(data=self.data, threshold=-0.3) == 0))
        assert (len(gt.all_crossings(data=self.data, threshold=3.5) == 0))