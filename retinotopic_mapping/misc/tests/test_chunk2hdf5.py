"""
test_hdf5tools.py

@author: derricw

Tests for hdf5tools.py

#TODO: use unittest module

"""

import os

import numpy as np
import h5py

from toolbox.misc import chunk2hdf5

def test_chunk2hdf5():
    """
    Tests the `chunk2hdf5` function with several data types and shapes.
    """
    sizes = [10000000, 10000100, 1000100]
    chunk_size = 10000000
    data_path = "temp_data"
    h5_path = "temp_data.h5"

    # test multiple data types
    data_types = [np.uint8, "<u2", np.int16, np.float32, np.float64]

    for tempfile in [data_path, h5_path]:
        if os.path.isfile(tempfile):
            os.remove(tempfile)

    for size in sizes:

        for dtype in data_types:

            data_1d = np.arange(0, size, 1, dtype=dtype)
            data_2d = data_1d.reshape((size/10, 10))
            data_3d = data_1d.reshape((size/10, 5, 2))
            data_4d = data_1d.reshape((size/100, 10, 5, 2))

            for data in [data_1d, data_2d, data_3d, data_4d]:
                with open(data_path, 'wb') as f:
                    f.write(np.ctypeslib.as_array(data))

                print data.shape

                chunk2hdf5(h5_path,
                           data_path,
                           dtype=dtype,
                           data_shape=data.shape,
                           data_name='data',
                           chunk_size=chunk_size)

                with h5py.File(h5_path, 'r') as hfile:
                    dset = hfile['data']
                    # check to ensure that what is in the h5 is the same
                    # as the original data
                    assert np.all(dset.value==data)

                os.remove(h5_path)

    os.remove(data_path)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    test_chunk2hdf5()
