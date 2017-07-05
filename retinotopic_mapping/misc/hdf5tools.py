"""
hdf5 tools

@author: derricw

AUG 14 2015

Test available at tests/test_chunk2hdf5.py


"""
import logging

import numpy as np
import h5py


def get_fileobj_size(file_obj):
    """
    Gets the size of an open file object.

    Args:
        file_obj (FileObject): an open file object.
        
    Returns:
        int: size of the file in bytes
    
    """
    start_pos = file_obj.tell()
    file_obj.seek(0, 2)
    size = file_obj.tell()
    file_obj.seek(start_pos)
    return size

def chunk2hdf5(h5_file,
               data_file,
               dtype,
               data_shape=(-1,),               
               data_name="data",
               header_bytes=0,
               chunk_size=10**9):
    """
    Loads the data into the hdf5 file in chunks.  Useful for really big
        binary files.
        
    Args:
        h5_file (h5py.File or str): An hdf5 file to add the data to.
        data_file (FileObject or str): A file containing a binary data set
        dtype (numpy.dtype): intended data type of binary data
        data_shape (Optional[tuple]): intended shape of binary data
        data_name (Optional[str]): name of destination dataset
        chunk_size (Optional[int]): maximum bytes per chunk. Default:10^9
        
    Returns:
        int: total bytes added to the hdf5 dataset.
        
    Raises:
        IOError: data file doesn't exist
        NameError: Dataset already has data by that name.
        IOError: Chunk shape not a multiple of row size.        
        
    """
    # get file objects if they gave us strings
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, 'a')
    if isinstance(data_file, str):
        data_file = open(data_file, 'rb')
        data_file.seek(header_bytes)
        
    if isinstance(data_shape, int):
        data_shape = (data_shape,)
        
    total_bytes = get_fileobj_size(data_file) - header_bytes
    itemsize = np.dtype(dtype).itemsize
    row_size = itemsize
    for l in data_shape[1:]:
        row_size *= l  
    
    total_rows = total_bytes / row_size    
    rows_per_chunk = chunk_size // row_size
    total_chunks = total_rows / rows_per_chunk
    if total_rows % rows_per_chunk != 0:
        total_chunks += 1
    rounded_chunk_size = rows_per_chunk * row_size
    rounded_chunk_items = rounded_chunk_size / itemsize
    maxshape = [None] + list(data_shape[1:])
    data_shape = [total_rows] + list(data_shape[1:])

    # create the dataset if it doesn't exist
    if data_name not in h5_file.keys():
        dset = h5_file.create_dataset(data_name,
                                      shape=data_shape,
                                      dtype=dtype,
                                      maxshape=maxshape)
    else:
        raise NameError("Dataset already exists! Choose a new name.")
        
    # add the data to the dataset
    sample_count = 0
    reshape = [-1]+ list(data_shape[1:])
    for chunk_count in range(total_chunks):
        #read a chunk
        chunk = np.fromfile(data_file,
                            dtype=dtype,
                            count=rounded_chunk_items,
                            )
        try:
            chunk = chunk.reshape(reshape)
        except ValueError:
            raise IOError("Chunk shape {} doesn't match desired shape. {}".format(chunk.shape, reshape))

        # if chunk has rows, write it.  if not break
        if chunk.shape[0] > 0:
            start = chunk_count * rows_per_chunk
            stop = start + chunk.shape[0]            
            dset[start:stop] = chunk
            sample_count+=chunk.size
            print("Moved {} out of {} chunks.".format(chunk_count+1, total_chunks))
        else:
            logging.info("chunk2hdf5 Finished!")
            break
    logging.info(" - moved {} samples in {} chunks.".format(sample_count,
                                                            chunk_count))
    h5_file.close()
    data_file.close()
    return sample_count
