__version__ = '2.6.0'

def test():
    import pytest
    import os

    curr_file_path = os.path.realpath(__file__)
    curr_file_dir = os.path.dirname(curr_file_path)
    test_dir = os.path.join(curr_file_dir, 'test')
    test_dir = test_dir.replace('\\', '/')
    pytest.main(test_dir)