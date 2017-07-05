"""

slicer.py

@author: derricw

Allen Institute for Brain Science.

"""
import os
import numpy as np


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray


class BinarySlicer(object):
    """
    Slices a binary file (up to 3 axes) like an ndarray, without reading the
        whole file into memory.

    It has many of the same attributes as an ndarray, like ndim, dtype, shape,
        etc.

    THINGS THAT DONT WORK YET
    1. np.newaxis (or for example indexing with "..." )
    2. array indexing (indexing using an arbitrary array of indices)
    3. logical indexing (indexing using a bool array)
    4. Dimensions higher than 3 (for example color image stacks, etc)
        (however you can use reshaping tricks to accomplish this)
    5. Using -1 in your shape argument to guess a dimension length

    Parameters
    ----------
    file_like : str or FileObject
        File to read from.
    shape : tuple
        Shape to cast file into
    dtype : np.dtype
        Numpy data type to read the file as.  All slices will be returned as
        this type.
    memory_limit : int
        Set an arbitrary memory limit in bytes.  Any slices requested that are
        larger than this value will raise an exception.
    header : int
        Header size in bytes.

    Examples
    --------

    >>> bs = BinarySlicer('my_binary_file', shape=(1000, 480, 640))
    >>> first_frame = bs[0]
    >>> last_frame = bs[-1]
    >>> bs.close()

    >>> # or use a context manager
    >>> with BinarySlicer('my_binary_file', shape=(1000, 480, 640)) as f:
    ...     first_frame = bs[0]

    """
    def __init__(self,
                 file_like,
                 shape=None,
                 dtype=None,
                 memory_limit=10**9,
                 header=0):

        self.file_like = file_like
        self.shape = shape
        self.dtype = dtype
        self.memory_limit = memory_limit
        self.header = header

        if self.dtype:
            dt = np.dtype(self.dtype)
            self.itemsize = dt.itemsize
        else:
            self.itemsize = 1  # assume 1??
            self.dtype = np.uint8  # assume uint8??

        self._load(self.file_like)

    def _load(self, file_like):
        """
        Loads a file.
        """
        if isinstance(file_like, str):
            self._file = open(file_like, mode='rb')
            if os.path.splitext(file_like)[1] == ".npy":
                #if it is a numpy file we can get some info automagically
                #   from the numpy header
                self._file.seek(8)
                header_len = int(np.fromfile(self._file, count=1,
                                             dtype=np.uint16))
                header = eval(self._file.read(header_len))

                self.shape = header['shape']
                self.header = header_len + 10  # 8 + 2 + header_len
                self.dtype = np.dtype(header['descr'])
                self.itemsize = self.dtype.itemsize
                self.ndim = len(self.shape)

        elif isinstance(file_like, file):
            self._file = file_like

        self._file.seek(0, 2)  # end of file
        self.nbytes = self._file.tell() - self.header
        self.size = self.nbytes / self.itemsize
        self._file.seek(0)

        if not self.shape:
            self.shape = (self.size,)

        self.ndim = len(self.shape)

        #check to see if shape/size agree
        s = self.itemsize
        for dim in self.shape:
            s *= dim
        if self.nbytes == s:
            pass
        else:
            print(" WARNING: File size %s (with header %s) into shape %s" % (self.nbytes,
                self.header, str(self.shape)))

    def __getitem__(self, index):
        """
        Method called when object is indexed.
        """
        if isinstance(index, int):
            return self._get_slice_int(index)
        elif isinstance(index, slice):
            return self._get_slice_slice(index)
        elif isinstance(index, tuple):
            return self._get_slice_tuple(index)
        elif index is Ellipsis:
            return self[:]
        else:
            raise TypeError("Index must be int or slice or tuple of slices.")

    def _get_slice_int(self, index):
        """
        Gets a slice by integer.

        """
        if index < 0:
            index = self.shape[0] + index

        file_pos = self._get_row_position(index)
        slice_size = self._get_row_size()

        if slice_size*self.itemsize > self.memory_limit:
            raise ValueError("Slice size greater than memory limit.")

        self._file.seek(file_pos + self.header)

        data = np.fromfile(self._file, count=slice_size, dtype=self.dtype)

        data = data.reshape(self._get_slice_shape(index))

        return data

    def _get_slice_slice(self, index):
        """
        Get slice by slice.

        """
        start, stop, step = self._condition_slice(index, 0)
        index = slice(start, stop, step)

        start_pos = self._get_row_position(start)

        total_rows = self._get_row_count(index)

        row_size = self._get_row_size()
        slice_size = row_size * total_rows

        if slice_size*self.itemsize > self.memory_limit:
            raise ValueError("Slice size exceeds memory limit.")

        npy_slice = np.zeros(slice_size, dtype=self.dtype)
        for i in range(total_rows):
            f_pos = start_pos + i*row_size*step*self.itemsize + self.header
            self._file.seek(f_pos)
            npy_slice[i*row_size:(i+1)*row_size] = np.fromfile(self._file,
                count=row_size, dtype=self.dtype)

        slice_shape = self._get_slice_shape(index)

        npy_slice = np.reshape(npy_slice, slice_shape)

        return npy_slice

    def _get_slice_tuple(self, index):
        """
        For fancy indexing.
        """
        # condition our slices, extract indices
        if len(index) == 2:
            index = (index[0], index[1], slice(0, self.shape[2], 1))

        f_start, f_stop, f_step = self._condition_slice(index[0], 0)
        y_start, y_stop, y_step = self._condition_slice(index[1], 1)
        x_start, x_stop, x_step = self._condition_slice(index[2], 2)

        frame_index = slice(f_start, f_stop, f_step)
        y_index = slice(y_start, y_stop, y_step)
        x_index = slice(x_start, x_stop, x_step)

        # some values we need
        number_of_frames = self._get_row_count(frame_index)
        number_of_rows = self._get_row_count(y_index)
        number_of_cols = self._get_row_count(x_index)

        #total_slices size
        total_size = number_of_frames * number_of_rows * number_of_cols * self.itemsize
        if total_size > self.memory_limit:
            raise ValueError("Slice size greater than memory limit.")

        #slice size of each component
        frame_size = self._get_row_size(0)
        y_size = self._get_row_size(1)
        x_size = self._get_row_size(2)

        #start position of each component
        frame_start_position = frame_size * f_start * self.itemsize
        y_start_position = y_size * y_start * self.itemsize
        x_start_position = x_size * x_start * self.itemsize

        #start position of each read operation
        frame_positions = [frame_start_position + frame_size * i * f_step * self.itemsize for i in range(number_of_frames)]
        if f_step < 0:
            frame_positions = [abs(f) for f in frame_positions[::-1]]
        y_positions = [y_start_position + y_size * i * y_step * self.itemsize for i in range(number_of_rows)]
        if y_step < 0:
            y_positions = [abs(y) for y in y_positions[::-1]]
        x_positions = [x_start_position + x_size * i * x_step * self.itemsize for i in range(number_of_cols)]
        if x_step < 0:
            x_positions = [abs(x) for x in x_positions[::-1]]

        #output array
        output = np.zeros((number_of_frames, number_of_rows, number_of_cols),
                          dtype=self.dtype)

        #read all positions
        for f in range(number_of_frames):
            for y in range(number_of_rows):
                fpos = frame_positions[f] + y_positions[y] + x_positions[0] + self.header
                self._file.seek(fpos)

                output[f, y, :] = np.fromfile(self._file, dtype=self.dtype,
                    count=number_of_cols*abs(x_step))[::x_step]

        #replicate numpy output behavior
        if number_of_frames == 1:
            output = output[0]

        return output

    def _condition_slice(self, indices, dimension):
        """
        Covers a lot of edge cases for slicing.

        """
        #Handle integer slices (convert it to a slice)
        if isinstance(indices, int):
            if indices < 0:
                i = int(self.shape[dimension]) + indices
                indices = slice(i, i+1, 1)
            else:
                indices = slice(indices, indices+1, 1)
        if indices is Ellipsis:
            indices = slice(0, int(self.shape[dimension]), 1)

        #Extract positions
        #print indices
        start, stop, step = indices.start, indices.stop, indices.step
        #print start, stop, step

        #Handle step possibilities
        if step is None:
            step = 1

        #handle start possibilities
        if start is None:
            if step > 0:
                start = 0
            elif step < 0:
                start = int(self.shape[dimension]) - 1
        elif start < 0:
            start = int(self.shape[dimension]) + start
        elif start >= 0:
            pass

        #handle stop possibilities
        if stop is None:
            if step > 0:
                stop = int(self.shape[dimension])
            elif step < 0:
                stop = -1
            else:
                raise RuntimeError("Here there be dragons.")
        elif stop < 0:
            stop = int(self.shape[dimension]) + stop
        elif stop >= 0:
            pass

        start, stop, step = int(start), int(stop), int(step)

        #handle impossible indexes
        if step < 0:
            if stop > start:
                raise RuntimeError("Start before stop with negative step.")
        elif step > 0:
            if stop < start:
                raise RuntimeError("Start after stop with positive step.")

        #print start, stop, step
        #raise RuntimeError("\nFinished...")
        return start, stop, step

    def _get_row_count(self, index):
        """
        Returns the number of rows.
        """
        start, stop, step = index.start, index.stop, index.step
        if stop is None:  # handle case of [::-1]
            temp_stop = -1
        else:
            temp_stop = stop
        if step:
            rows = abs(int(temp_stop-start)/step)
            if (temp_stop-start) % step != 0:
                return rows + 1
                # if step > 0:
                #     return rows + 1
                # else:
                #     return rows
            else:
                return rows
        else:
            return int(temp_stop-start)

    def _get_row_size(self, dim=0):
        """
        Gets the size of a slice at a specified index.
        """
        slice_size = self.shape[(dim+1):]
        size = 1  # always?
        for s in slice_size:
            size *= s
        return size

    def _get_slice_shape(self, index):
        """
        Returns the shape of a slice.
        """
        if isinstance(index, int):
            return self.shape[1:]
        if isinstance(index, slice):
            rows = self._get_row_count(index)
            shape = [rows]
            shape.extend(self.shape[1:])
            return shape

    def _get_row_position(self, index):
        """
        Returns the position of a index on axis 0.
        """
        slice_size = self._get_row_size()
        pos = slice_size * index * self.itemsize
        return pos

    def __del__(self):
        """
        Close file when object is cleaned up.
        """
        self._file.close()

    def __enter__(self):
        """
        So we can use context manager (with...as) like any other open file.

        Examples
        --------
        >>> with BinarySlicer('my_data') as d:
        ...     d[0]

        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Exit statement for context manager.
        """
        self._file.close()


if __name__ == '__main__':
    # stack = np.arange(0, 1000, 1, dtype=np.uint16).reshape((10, 10, 10))
    # print stack[..., 0]
    # stack.tofile('test_stack')

    # bs = BinarySlicer('test_stack', shape=(10, 10, 10), dtype=np.uint16)

    # print bs[:]
    pass
