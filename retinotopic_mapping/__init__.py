import os
__version__ = '2.9.1'

def test():
    import pytest
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(curr_dir, 'test')
    test_dir = test_dir.replace('\\', '/')
    pytest.main(test_dir)