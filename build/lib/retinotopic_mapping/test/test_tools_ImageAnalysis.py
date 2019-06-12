import os
import unittest
import retinotopic_mapping.tools.ImageAnalysis as ia

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestSimulation(unittest.TestCase):

    def setUp(self):
        pass

    def test_distance(self):
        assert (ia.distance(3., 4.) == 1.)
        assert (ia.distance([5., 8.], [9., 11.]) == 5.)