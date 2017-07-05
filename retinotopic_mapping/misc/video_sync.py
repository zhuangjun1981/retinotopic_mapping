import cv2
import numpy as np
import h5py
import sys

def extract_embedded_avi(hdf):
    """
    Extract an embedded avi file from an hdf5.

    This function is for use with the hdf5 files generated from the
    video monitoring software, which don't store the movies as a frame
    dataset and instead store an encoded avi file within the hdf5.

    Args:
        hdf : (string)
            Filename of hdf5 file, terminating in .h5.

    Returns:
        avi : (string)
            Filename of the extracted avi.

    """
    with h5py.File(hdf, 'r') as f:
        ds = f['movie']
        vid = ds.value
        pad = int(str(ds.dtype).lstrip('S|')) - int(str(vid.dtype).lstrip('S|'))
    avi = hdf.replace('.h5', '.avi')
    with open(avi, 'wb') as f:
        f.write(vid + b'\x00'*pad)
    return avi

class SyncedVideoCapture(object):
    """
    Video capture class for synchronizing output.

    """
    def __init__(self, video, frame_times, max_delay=1.0):
        super(SyncedVideoCapture, self).__init__()
        """
        Constructor.

        Create the synchronized video capture object.

        Args:
            video : (string or numpy.ndarray/hdf5 dataset)
                Either the name of an avi file or an array or hdf5
                dataset containing frame information.
            frame_times : (numpy.ndarray)
                Array of timestamps for each frame in seconds. Must
                be the same length as as the number of frames in
                the video.
            max_delay : (float)
                Maximum time from last found frame in this video
                before blank frames should be displayed.

        Raises:
            AttributeError, IndexError

        """
        # frame_times should be a ndarray of same length as number of frames
        self.frame_times = frame_times
        self.max_delay = max_delay
        self.idx = 0
        if isinstance(video, str):
            # If video is a string, assume it is a filename for an avi
            self._vid = cv2.VideoCapture(video)
            N_frames = self._vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            if self.frame_times.shape[0] != N_frames:
                raise AttributeError('Video length does not match time array.')
            setattr(self, 'get_frame', self._get_frame_movie)
            rc, self.frame = self._vid.read()
            if not rc:
                raise IndexError('Unable to read initial frame of %s.' % video)
            self.blank = np.zeros(self.frame.shape, dtype=self.frame.dtype)
        else:
            # Otherwise assume it is an opened hdf5 dataset or numpy array
            # assume the time axis is the 0 axis of the array
            self._vid = video
            if self.frame_times.shape[0] != self._vid.shape[0]:
                raise AttributeError('Video length does not match time array.')
            setattr(self, 'get_frame', self._get_frame_numpy)
            self.frame = self._vid[0]
            self.blank = np.zeros(self.frame.shape, dtype=self.frame.dtype)

    def _get_frame_numpy(self, t):
        """
        Get frame at time t from a dataset or ndarray.

        Args:
            t : (float)
                Time for which to grab the corresponding frame.

        Returns:
            frame : (numpy.ndarray)
                Frame at time t.

        """
        pre_arr = np.where(self.frame_times <= t)[0]
        if pre_arr.size > 0:
            idx = pre_arr[-1]
        else:
            return self.blank
        if t - self.frame_times[idx] > self.max_delay:
            return self.blank
        return self._vid[idx]

    def _get_frame_movie(self, t):
        """
        Get frame at time t from an avi.

        Args:
            t : (float)
                Time for which to grab the corresponding frame.

        Returns:
            frame : (numpy.ndarray)
                Frame at time t.

        """
        pre_arr = np.where(self.frame_times <= t)[0]
        if pre_arr.size > 0:
            idx = pre_arr[-1]
        else:
            return self.blank
        if idx > self.idx:
            self.idx = idx
            self._vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, idx)
            rc, self.frame = self._vid.read()
            if not rc:
                raise IndexError('Unable to read frame %s.' % self.idx)
        if t - self.frame_times[idx] > self.max_delay:
            return self.blank
        return self.frame

    def height(self):
        """
        Return the height of a frame.

        """
        return self.blank.shape[0]

    def width(self):
        """
        Return the width of a frame.

        """
        return self.blank.shape[1]

    def release(self):
        """
        Release the underlying video stream.

        """
        try:
            self._vid.release()
        except Exception as e:
            print e

class MovieCombiner(object):
    """
    Class to combine different video streams in a time-synced manner.

    """
    def __init__(self, subsample=True, out_color=False):
        """
        Constructor.

        Args:
            subsample : (bool)
                Whether or not to subsample. If true, output is
                subsampled 2x2.
            out_color : (bool)
                Whether or not output should be color.

        """
        super(MovieCombiner, self).__init__()
        self.subsample = subsample
        self.out_color = out_color
        self._shape = [0,0]
        self._streams = {}
        self._primary = None

    def add_stream(self, synced_capture, column, row, frame_callback=None,
                   primary=False):
        """
        Add a video stream to the combiner.

        Args:
            synced_capture : (SyncedVideoCapture)
                Synchronized video capture object.
            column : (int)
                Column in which the stream should be in the output.
            row : (int)
                Row in which the stream should be in the output.
            frame_callback : (callable)
                Function to call which takes a frame and returns a frame
                of the same shape after doing some work.
            primary : (bool)
                Whether or not this is the primary stream. The first
                stream added defaults to primary. The primary stream
                determines the basis timescale for the output video.

        Raises:
            AttributeError
        """
        if self._streams.get((column, row), False):
            raise AttributeError('Stream already assigned to (%s, %s)' %
                                 (column, row))
        self._streams[(column, row)] = (synced_capture, frame_callback)
        if column >= self._shape[0]:
            self._shape[0] = column + 1
        if row >= self._shape[1]:
            self._shape[1] = row + 1
        if (self._primary is None) or primary:
            self._primary = (column, row)
        
    def write_movie(self, fname, output_callback=None):
        """
        Write an output movie.

        Args:
            fname : (string)
                Name of the output movie.
            output_callback : (callable)
                Function which takes an output frame and returns
                a frame of the same shape after doing some work to
                it.

        """
        d = 1
        if self.subsample:
            d = 2
        self._frame_def()
        times = self._streams[self._primary][0].frame_times
        fr = 1/np.mean(np.diff(times))
        cv_shape = (self.frame_shape[1], self.frame_shape[0])
        writer = cv2.VideoWriter(fname, cv2.cv.CV_FOURCC(*'XVID'), fr,
                                 cv_shape, self.out_color)
        for i in xrange(times.shape[0]):
            frame = np.zeros(self.frame_shape, dtype=np.ubyte)
            sys.stdout.write('\rProcessing %d of %d' % (i+1, times.shape[0]))
            for k, v in self._streams.iteritems():
                c = k[0]
                r = k[1]
                s = v[0]
                cb = v[1]
                tmp = s.get_frame(times[i])[::d,::d]
                if cb:
                    tmp = cb(tmp)
                xo = np.sum(self._x_offsets[:c+1])
                yo = np.sum(self._y_offsets[:r+1])
                if self.out_color and (tmp.ndim == 2):
                    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
                elif (not self.out_color) and (tmp.ndim > 2):
                    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                frame[yo:yo+s.height()/d,xo:xo+s.width()/d] = tmp
            if output_callback:
                frame = output_callback(frame)
            writer.write(frame)

    def _frame_def(self):
        """
        Helper function to define frame geometry.

        """
        d = 1
        if self.subsample:
            d = 2
        self._x_offsets = np.zeros(self._shape[0]+1, dtype=np.int)
        self._y_offsets = np.zeros(self._shape[1]+1, dtype=np.int)
        for k, v in self._streams.iteritems():
            c = k[0]
            r = k[1]
            s = v[0]
            if s.width()/d > self._x_offsets[c+1]:
                self._x_offsets[c+1] = s.width()/d
            if s.height()/d > self._y_offsets[r+1]:
                self._y_offsets[r+1] = s.height()/d
        self.frame_shape = (np.sum(self._y_offsets), np.sum(self._x_offsets))
        if self.out_color:
            self.frame_shape = self.frame_shape + (3,)

    def release_streams(self):
        """
        Release all of the underlying video streams.

        """
        for k, v in self._streams.iteritems():
            v[0].release()

def frame_times_from_duration(vid, duration):
    v = cv2.VideoCapture(vid)
    N = v.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    v.release()
    return np.arange(N)*(duration/N)