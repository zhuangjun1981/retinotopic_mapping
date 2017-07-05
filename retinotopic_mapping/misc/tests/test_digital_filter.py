import numpy as np
import unittest
from toolbox.misc.ophystools import filter_digital

class TestFilter(unittest.TestCase):
    def test_filter(self):

        # test with no transients
        original_r = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
        original_f = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
        normal_r, normal_f = filter_digital(original_r, original_f)

        # test with transients (low-to-high)
        test_r = np.array([0.0, 0.02, 0.04, 0.04501, 0.06, 0.08])
        test_f = np.array([0.01, 0.03, 0.04500, 0.05, 0.07, 0.09])
        l2h_r, l2h_f = filter_digital(test_r, test_f)

        # test with transients (high-to-low)
        test_r = np.array([0.0, 0.02, 0.03500, 0.04, 0.06, 0.08])
        test_f = np.array([0.01, 0.03, 0.03501, 0.05, 0.07, 0.09])
        h2l_r, h2l_f = filter_digital(test_r, test_f)

        # test with both
        test_r = np.array([0.0, 0.02, 0.03500, 0.04, 0.04501, 0.06, 0.08])
        test_f = np.array([0.01, 0.03, 0.03501, 0.04500, 0.05, 0.07, 0.09])
        combined_r, combined_l = filter_digital(test_r, test_f)

        # all results should be the same as the original rising and falling after filtering
        self.assertTrue(original_r.all() == normal_r.all() == l2h_r.all() == h2l_r.all() == combined_r.all())
        self.assertTrue(original_f.all() == normal_f.all() == l2h_f.all() == h2l_f.all() == combined_l.all())

    def test_errors(self):
        # test with array size mismatch
        original_r = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
        original_f = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
        self.assertRaises(ValueError, filter_digital, original_r, original_f)

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(TestFilter)
    unittest.TextTestRunner(verbosity=2).run(suite)