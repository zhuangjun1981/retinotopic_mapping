################################################################################
# Copyright 2015, The Allen Institute for Brain Science
# Author: Jed Perkins
################################################################################
import os
import sys
import time
import shutil
import stat
import hashlib

DEFAULT_BUFFER=16*1024*1024
SHA1_CHUNK = hashlib.sha1().block_size*4096

def copyfileobj_cb(srcf, dstf, callback=None, buf_size=DEFAULT_BUFFER):
    """
    Same as shutil.copyfileobj, but with a callback every iteration.

    Args:
        srcf : file object
            Source file object.
        dstf : file object
            Destination file object.
        callback : callable(buffer length)
            Callback to call every iteration of data transfer.
        buf_size : int
            Buffer size. Defaults to `DEFAULT_BUFFER`

    """
    while True:
        buf = srcf.read(buf_size)
        if not buf:
            break
        dstf.write(buf)
        if callback:
            callback(len(buf))

def copyfile_cb(src, dst, callback):
    """
    Same as shutil.copyfile, but with a callback called iteratively.

    Args:
        src : string
            Source filename.
        dst : string
            Destination filename.
        callback : callable(buffer length)
            Callback to call each iteration of data transfer.

    """
    if shutil._samefile(src, dst):
        raise shutil.Error("`%s` and `%s` are the same file" % (src, dst))

    for fn in [src, dst]:
        try:
            st = os.stat(fn)
        except OSError:
            pass
        else:
            # Handle special files (sockets, devices...)
            if stat.S_ISFIFO(st.st_mode):
                raise shutil.SpecialFileError("`%s` is a named pipe" % fn)

    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            copyfileobj_cb(fsrc, fdst, callback)

def copy_cb(src, dst, callback):
    """
    Same as shutil.copy, but with a callback called iteratively.

    Args:
        src : string
            Source filename.
        dst : string
            Destination directory or filename.
        callback : callable(buffer length)
            Callback to call each iteration of data transfer.

    """
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    copyfile_cb(src, dst, callback)
    shutil.copymode(src, dst)

class ProgressTracker(object):
    """
    Simple progress tracker.

    Args:
        total : float
            Total value to track progress out of.
    """
    def __init__(self, total=100.0):
        """
        Constructor.

        """
        super(ProgressTracker, self).__init__()
        self._prog = 0
        self.set_total(total)

    def get_progress(self, increment):
        """
        Return the progress given the current increment.

        Args:
            increment : float
                Incremental addition to progress.

        Returns:
            progress : float
                Progress as the fraction sum(increments)/total.

        """
        self._prog += increment
        if self.total:
            return float(self._prog)/self._total
        else:
            return self._prog

    def set_total(self, total):
        """
        Set the total value to track progress out of.

        Args:
            total : float
                Total value to track progress out of.

        """
        self._total = total

    def reset(self):
        """
        Resety the progress tracker.

        """
        self._prog = 0

class RateTracker(object):
    """
    Simple rate tracker.

    """
    def __init__(self):
        """
        Constructor.

        """
        super(RateTracker, self).__init__()
        self._last_t = None
        self._init_t = None

    def get_rate(self, size):
        """
        Get the rate given the size.

        Args:
            size : float
                Size of the object.

        Returns:
            rate : float
                Rate as size/s.

        """
        t = time.clock()
        if self._last_t is None:
            self._init_t = t
            rate = 0
        else:
            rate = float(size)/(t-self._last_t)
        self._last_t = t
        return rate

    def reset(self):
        """
        Reset the rate tracker.

        """
        self._last_t = None
        self._init_t = None

class FileCopy(object):
    """
    Class for handling file copying with optional rate and progress
    tracking.

    Args:
        track_progress : bool
            Whether to track progress.
        track_rate : bool
            Whether to track rate.
        track_callback : callable(progess, rate)
            Callback to call for displaying progress and rate.

    """
    def __init__(self, track_progress=True, track_rate=True,
                 track_callback=None):
        """
        Constructor.

        """
        self.pt = False
        self.rt = False
        if track_progress:
            self.pt = ProgressTracker()
        if track_rate:
            self.rt = RateTracker()
        if track_callback is None:
            self.track_callback = self._display_progress
        else:
            self.track_callback = track_callback

    def copy(self, src, dst):
        """
        Copy file while displaying information about rate/progress.

        Args:
            src : string
                Source file to copy.
            dst : string
                Destination file or directory.

        """
        if shutil._samefile(src, dst):
            raise shutil.Error("`%s` and `%s` are the same file" % (src, dst))

        sz = None
        for fn in [src, dst]:
            try:
                st = os.stat(fn)
                sz = st.st_size
            except OSError as e:
                print e
            else:
                # Handle special files (sockets, devices...)
                if stat.S_ISFIFO(st.st_mode):
                    raise shutil.SpecialFileError("`%s` is a named pipe" % fn)
            if (fn == src) and (self.pt):
                self.pt.set_total(sz)
        copy_cb(src, dst, self._copy_callback)

    def _display_progress(self, progress, rate):
        """
        Display information about rate/progress to stdout.

        Args:
            progress : float
                Progress as a fraction (0 to 1).
            rate : float
                Rate measured in bytes/s.

        """
        pstr = ""
        rstr = ""
        if progress:
            if progress == 1.0:
                sys.stdout.write("\rFile copy complete!\n")
                sys.stdout.flush()
                return
            else:
                pstr = ("%.1f%%" % (100*progress)).ljust(6)
        if rate:
            rstr = self._rate_string(rate).rjust(12)
        sys.stdout.write("\r%s %s" % (pstr, rstr))
        sys.stdout.flush()

    def _copy_callback(self, inc):
        """
        Callback to call on copy to track progress and rate.

        """
        if self.pt and self.pt.total:
            prog = self.pt.get_progress(inc)
        else:
            prog = None
        if self.rt:
            rate = self.rt.get_rate(inc)
        else:
            rate = None
        self.track_callback(prog, rate)

    def _rate_string(self, rate):
        """
        Convert bytes/s rate to a nice display string.

        Args:
            rate : float
                Rate in bytes/s.

        Returns:
            rate_str : string
                Display string for rate.

        """
        if rate > 1024**2:
            return '%.2f MB/s' % (rate/1024**2)
        elif rate > 1024**2:
            return '%.2f KB/s' % (rate/1024)
        else:
            return '%.2f B/s' % rate

def fancy_copy(src, dst):
    """
    Copy file with rate and progress tracker to stdout.

    Args:
        src : string
            Source file to copy.
        dst : string
            Destination file or directory.

    """
    fc = FileCopy()
    fc.copy(src, dst)

def hash_file(filename, algorithm='sha1', chunk_size=SHA1_CHUNK):
    """
    Hash a file with any algorithm supported by the Python
    distribution.

    Args:
        filename : string
            The name of the file to be hashed.
        algorithm : string
            The hashing algorithm to use.
        chunk_size : int
            Block size to use when reading file.

    Returns:
        hex_hash : string
            The hex hash of the digested file.
    """
    alg_inst = hashlib.new(algorithm)
    with open(fname, 'rb') as f:
        chunk = f.read(chunk_size)
        while chunk:
            alg_inst.update(chunk)
            chunk = f.read(chunk_size)
    return alg_inst.hexdigest()

def check_file_hashes(file_list, hash_dict, algorithm='sha1'):
    """
    Checks a list of files against a set of hashes and returns a list
    of any non-matching files.

    Args:
        file_list : list
            List of files to check.
        hash_dict : dict
            Dictionary with keys of filenames and values of associated
            file hashes.
        algorithm : string
            The hashing algorithm the hashes were generated with.

    Returns:
        nonmatching_file : list
            All files that don't match the expected hash.
    """
    nonmatching = []
    for filename in file_list:
        basename = os.path.basename(filename)
        real_hash = hash_file(filename, algorithm)
        if real_hash != hash_dict[basename]:
            nonmatching.append(filename)
    return nonmatching


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