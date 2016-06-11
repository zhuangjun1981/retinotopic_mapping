__author__ = 'junz'

def test():
    import pytest
    import os
    import inspect

    curr_file_name = inspect.getfile(inspect.currentframe())
    curr_file_dir = os.path.dirname(curr_file_name)
    test_dir = os.path.join(curr_file_dir, 'test')
    test_dir = test_dir.replace('\\', '/')
    pytest.main(test_dir)