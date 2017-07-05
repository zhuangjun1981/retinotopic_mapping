# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:05:33 2016

@author: derricw

A few tools for analyzing an ophys experiment.

"""
import os
import pickle
import json
import collections
import traceback

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt

from sync import Dataset


def filter_digital(rising, falling, threshold=0.0001):
    """
    Removes short transients from digital signal.
    
    Rising and falling should be same length and units
        in seconds.

    Kwargs:
        threshold (float): transient width
    """
    # forwards (removes low-to-high transients)
    dif_f = falling-rising
    falling_f = falling[dif_f > threshold]
    rising_f = rising[dif_f > threshold]
    # backwards (removes high-to-low transients )
    dif_b = rising_f[1:]-falling_f[:-1]
    dif_br = np.append([threshold*2],dif_b)
    dif_bf = np.append(dif_b, [threshold*2])
    rising_f = rising_f[np.abs(dif_br) > threshold]
    falling_f = falling_f[np.abs(dif_bf) > threshold]
    
    return rising_f, falling_f

def add_data_point(data_list, sample_no, bit, value):
    """ Inserts an event on a specified bit at a specified sample number
    """
    events = [d[0] for d in data_list]  # sample no for each event
    prev_index = [n for n,i in enumerate(events) if i < sample_no][-1]

    prev_value = data_list[prev_index][1]
    bit_value = 2**bit
    # check bit
    if (np.bitwise_and(prev_value, bit_value) > 0) and (value == 1):
        raise ValueError("Bit already high at this sample.")
    elif(np.bitwise_and(prev_value, bit_value) == 0) and (value == 0):
        raise ValueError("Bit already low at this sample.")
    elif (np.bitwise_and(prev_value, bit_value) > 0) and (value == 0):
        new_value = prev_value - bit_value
    elif (np.bitwise_and(prev_value, bit_value) == 0) and (value == 1):
        new_value = prev_value + bit_value
        
    data_list.insert(prev_index+1, [sample_no, new_value])


def set_bits_in_range(data_list, bit, start, stop, value):
    """ Sets a bit to a value for all events between start:stop
    """
    events = [d[0] for d in data_list]
    prev_index = [n for n,i in enumerate(events) if i < start][-1]
    post_index = [n for n, i in enumerate(events) if i > stop][0]
    
    for i, d in enumerate(data_list):
        if prev_index < i <= post_index:
            if value == 1:
                data_list[i][1] = bit_high(data_list[i][1], bit)
            elif value == 0:
                data_list[i][1] = bit_low(data_list[i][1], bit)
            else:
                raise ValueError("Bit value needs to be 0 or 1")
            
def bit_high(integer, bit):
    """ Sets a bit in an integer to 1 """
    return (integer | (1 << bit))
    
def bit_low(integer, bit):
    """ Sets a bit in an integer to 0 """
    return (integer & ~(1 << bit))


class Dataset2p(Dataset):
    """
    Extends the sync.Dataset class to include some 2p-experiment-specific
        functions.  Stores a cache of derived values so subsequent calls can
        be faster.

    Args:
        path (str): path to sync file
    
    Example:
    
        >>> dset = Dataset2p("sync_data.h5")
        >>> dset.display_lag
        0.05211234
        >>> dset.stimulus_start
        31.8279931238
        >>> dset.plot_start()  # plots the start of the experiment
        >>> dset.plot_end() # plots the end fo the experiment
        >>> dset.plot_timepoint(31.8)  # plots a specific timepoint
        
    """
    def __init__(self, path):
        super(Dataset2p, self).__init__(path)
        
        self._cache = {}
        
    def signal_exists(self, line_name):
        """
        Checks to see if there are any events on a specified line.
        """
        if len(self.get_events_by_line(line_name)) > 0:
            return True
        else:
            return False
            
    @property
    def sample_freq(self):
        """
        The frequency that the sync sampled at.
        """
        return self.meta_data['ni_daq']['counter_output_freq']            
            
    @property
    def display_lag(self):
        """
        The display lag in seconds.  This is the latency between the display
            buffer flip pushed by the video card and buffer actually being
            drawn on screen.
        """
        #pd0 = self.stimulus_start
        #vs0 = self.get_stim_vsyncs()[0]
        #return pd0-vs0
        stim_vsyncs = self.get_stim_vsyncs()
        transitions = stim_vsyncs[::60]
        photodiode_events = self.get_real_photodiode_events()[0:len(transitions)]
        return np.mean(photodiode_events-transitions)
        
        
    @property
    def stimulus_start(self):
        """
        The start of the visual stimulus, accounting for display lag.
        """
        return self.get_photodiode_events()[0]
        
    @property
    def stimulus_end(self):
        """
        The end of the visual stimulus, accounting for display lag.
        """
        vs_end = self.get_stim_vsyncs()[-1]
        return vs_end + self.display_lag
        
    @property
    def stimulus_duration(self):
        """
        The duration of the visual stimulus.
        """
        return self.stimulus_end - self.stimulus_start
        
    @property
    def twop_start(self):
        """
        The start of the two-photon acquisition.
        """
        return self.get_twop_vsyncs()[0]
        
    @property
    def twop_end(self):
        """
        The start of the two-photon acquisition.
        """
        return self.get_twop_vsyncs()[-1]
        
    @property
    def twop_duration(self):
        return self.twop_end - self.twop_start
        
    @property
    def video_duration(self):
        return [v[-1] - v[0] for v in self.get_video_vsyncs()]
        
    def get_long_stim_frames(self, threshold=0.025):
        """
        Get dropped frames for the visual stimulus using a duration threshold.
        
        Args:
            threshold (float): minimum duration in seconds of a "long" frame.
        """
        vsyncs = self.get_stim_vsyncs()
        vs_intervals = self.get_stim_vsync_intervals()
        drop_indices = np.where(vs_intervals > threshold)
        drop_intervals = vs_intervals[drop_indices]
        drop_times = vsyncs[drop_indices]  # maybe +1???
        return {'indices': drop_indices,
                'intervals': drop_intervals,
                'times': drop_times}
                
    def get_long_twop_frames(self, threshold=0.040):
        """
        Get dropped frames for the two photon using a duration threshold.
        
        Args:
            threshold (float): minimum duration in seconds of a "long" frame.
        """
        vsyncs = self.get_twop_vsyncs()
        vs_intervals = self.get_twop_vsync_intervals()
        drop_indices = np.where(vs_intervals > threshold)
        drop_intervals = vs_intervals[drop_indices]
        drop_times = vsyncs[drop_indices]  # maybe +1???
        return {'indices': drop_indices,
                'intervals': drop_intervals,
                'times': drop_times}
        
    def clear_cache(self):
        """
        Clears the cache of derived values.
        """
        self._cache = {}
        
    def get_photodiode_events(self):
        """
        Returns the photodiode events with the start/stop indicators and the
            window init flash stripped off.
        """
        if 'pd_events' in self._cache:
            return self._cache['pd_events']
        
        pd_name = 'stim_photodiode'        
        
        all_events = self.get_events_by_line(pd_name)        
        pdr = self.get_rising_edges(pd_name)
        pdf = self.get_falling_edges(pd_name)
        
        all_events_sec = all_events/self.sample_freq
        pdr_sec = pdr/self.sample_freq
        pdf_sec = pdf/self.sample_freq
        
        pdf_diff = np.ediff1d(pdf_sec, to_end=0)
        pdr_diff = np.ediff1d(pdr_sec, to_end=0)
        
        reg_pd_falling = pdf_sec[(pdf_diff >= 1.9) & (pdf_diff <= 2.1)]

        short_pd_rising = pdr_sec[(pdr_diff >= 0.1) & (pdr_diff <= 0.5)]
        
        first_falling = reg_pd_falling[0]
        last_falling = reg_pd_falling[-1]

        end_indicators = short_pd_rising[short_pd_rising > last_falling]
        first_end_indicator = end_indicators[0]
        
        pd_events =  all_events_sec[(all_events_sec >= first_falling) &
                                    (all_events_sec < first_end_indicator)]
        self._cache['pd_events'] = pd_events
        return pd_events
        
    def get_photodiode_anomalies(self):
        """
        Gets any anomalous photodiode events.
        """
        if 'pd_anomalies' in self._cache:
            return self._cache['pd_anomalies']

        events = self.get_photodiode_events()
        intervals = np.diff(events)
        anom_indices = np.where(intervals < 0.5)
        anom_intervals = intervals[anom_indices]
        anom_times = events[anom_indices]

        anomalies = {'indices': anom_indices,
                     'intervals': anom_intervals,
                     'times': anom_times,}

        self._cache['pd_anomalies'] = anomalies
        return anomalies
        
    def get_real_photodiode_events(self):
        """
        Gets the photodiode events with the anomalies removed.
        """
        events = self.get_photodiode_events()
        anomalies = self.get_photodiode_anomalies()['indices']
        return np.delete(events, anomalies)
                              
    def get_stim_vsyncs(self):
        """
        Returns the stimulus vsyncs in seconds, which is the falling edges of
            the 'stim_vsync' signal.
        """
        if 'stim_vsyncs' in self._cache:
            return self._cache['stim_vsyncs']
        
        sig_name = 'stim_vsync'     
        
        svs_r = self.get_rising_edges(sig_name)
        svs_f = self.get_falling_edges(sig_name)
        
        svs_r_sec = svs_r/self.sample_freq
        svs_f_sec = svs_f/self.sample_freq
        
        # Some versions of camstim caused a spike when the DAQ is first
        # initialized.  remove it if so.
        if svs_r_sec[1] - svs_r_sec[0] > 0.2:
            vsyncs = svs_f_sec[1:]
        else:
            vsyncs = svs_f_sec
            
        self._cache['stim_vsyncs'] = vsyncs
            
        return vsyncs
        
    def get_stim_vsync_intervals(self):
        return np.diff(self.get_stim_vsyncs())
        
    def get_twop_vsync_intervals(self):
        return np.diff(self.get_twop_vsyncs())
        
    def get_twop_vsyncs(self):
        """
        Returns the 2p vsyncs in seconds, which is the falling edges of the
            '2p_vsync' signal.
        """
        # this one is straight-forward
        return self.get_falling_edges('2p_vsync')/self.sample_freq
        
    def get_video_vsyncs(self):
        """
        Returns the video monitoring system vsyncs.
        """
        vsyncs = []
        for sync_signal in ['cam1_exposure', 'cam2_exposure']:
            falling_edges = self.get_falling_edges(sync_signal, units='sec')
            rising_edges = self.get_rising_edges(sync_signal, units='sec')
            rising, falling = filter_digital(rising_edges,
                                             falling_edges,
                                             threshold=0.000001)
            vsyncs.append(falling)
        return vsyncs
        
    def plot_timepoint(self,
                       time_sec,
                       width_sec=3.0,
                       signals=[],
                       out_file=""):
        """
        Plots signals around a specific timepoint, with adjustable
            width.

        Args:
            time_sec (float): time to plot at in seconds
            width_sec (float): width of the time range to plot
            signals (optional[list]): list of signals to plot
            
        """
        if not signals:
            # defaults
            signals = ['2p_vsync', 'stim_vsync', 'stim_photodiode',]
            
        start = time_sec - width_sec/2
        stop = time_sec + width_sec/2
        
        if out_file:
            auto_show = False
        else:
            auto_show = True
            
        fig =  self.plot_lines(signals, start_time=start, end_time=stop,
                               auto_show=auto_show)
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()

        return fig
        
    def plot_start(self, out_file=""):
        """
        Plots the start of the experiment.
        """
        start_time = self.stimulus_start
        return self.plot_timepoint(start_time, out_file=out_file)
        
    def plot_end(self, out_file=""):
        """
        Plots the end of the experiment.
        """
        end_time = self.stimulus_end
        return self.plot_timepoint(end_time, out_file=out_file)
        
    def plot_stim_vsync_intervals(self, out_file=""):
        """
        Plots the vsync intervals for the stimulus.
        """
        intervals = self.get_stim_vsync_intervals()
        plt.plot(intervals)
        plt.xlabel("frame number")
        plt.ylabel("duration (ms)")
        plt.title("Stimulus vsync intervals.")
        fig = plt.gcf()
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return fig
        
    def plot_twop_vsync_intervals(self, out_file=""):
        """
        Plots the vsync intervals for the two-photon data.
        """
        intervals = self.get_twop_vsync_intervals()
        plt.plot(intervals)
        plt.xlabel("frame number")
        plt.ylabel("duration (ms)")
        plt.title("2p vsync intervals.")
        fig = plt.gcf()
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return fig
        
    def plot_videomon_vsync_intervals(self, out_file=""):
        vsyncs = self.get_video_vsyncs()
        intervals = [np.diff(v) for v in vsyncs]
        subplots = len(intervals)
        f, axes = plt.subplots(subplots, sharex=True, sharey=True)
        if not isinstance(axes, collections.Iterable):
            axes = [axes]
        for data, ax in zip(intervals, axes):
            ax.plot(data)
        plt.xlabel("frame index")
        plt.ylabel("duration (ms)")
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return f
        
        
    def plot_stim_frame_hist(self, out_file=""):
        """
        Plots the visual stimulus frame histogram.  This is good for
            visualizing how "clean" your frame intervals are.
        """
        intervals = self.get_stim_vsync_intervals()
        plt.hist(intervals, bins=100, range=[0.016, 0.018])
        plt.xlabel("duration (sec)")
        plt.ylabel("frames")
        fig = plt.gcf()
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return fig
        
        
class OphysSession(object):
    """
    Single ophys session.  Provides some methods for loading the dataset
        and checking for anomalies.

    Loads the sync file, the pkl file, the video data, the video metadata, and
        the platform json file.        
        
    """
    def __init__(self, folder, other_data_folders=[]):
        super(OphysSession, self).__init__()
        
        self.folder = folder
        self.other_data_folders = other_data_folders

        if not os.path.isdir(self.folder):
            raise IOError("Experiment folder does not exist.")

        self.sync_data = None
        self._pkl_data = None
        self._video_data = None
        self._video_meta = None
        self._twop_data = None
        self._platform_data = {}
        
    def _find_path(self, match_str):
        """
        Finds a path in the experiment folder whose ending matches with
            `match_str`
        """
        
        # check primary data folder first
        matches = [fn for fn in os.listdir(self.folder) if fn.endswith(match_str)]
        if matches:
            return os.path.join(self.folder, matches[0])
        else:
            # if no matches, check other data folders for files with matching experiment ID
            exp_id = os.path.basename(self.folder).split("_")[0]  # format is EXPID_ANIMALNAME_TIMESTAMP
            for folder in self.other_data_folders:
                matches = [fn for fn in os.listdir(folder) if (fn.endswith(match_str) and fn.startswith(exp_id))]
                if matches:
                    return os.path.join(folder, matches[0])
        # no matches found, raise an error
        raise IOError("Couldn't find data file: {}".format(match_str))
        
        
    def load_sync(self, path=""):
        path = path or self._find_path("_sync.h5")
        self.sync_data = Dataset2p(path)
        
    def load_pkl(self, path=""):
        path = path or self._find_path("_stim.pkl")
        with open(path, "rb") as f:
            self._pkl_data = pickle.load(f)
            
    def load_videos(self,
                    eyetracking_path="",
                    behavior_path="",
                    eye_meta_path="",
                    behavior_meta_path="",):
        """
        TODO: make this more general in case they add more video streams.
        """

        eyetracking_path = eyetracking_path or self._find_path("_video-1.avi")
        behavior_path = behavior_path or self._find_path("_video-0.avi")
        eye_meta_path = eye_meta_path or self._find_path("_video-1.h5")
        behavior_meta_path = behavior_meta_path or self._find_path("_video-0.h5")
        
        self._video_data = [cv2.VideoCapture(path) for path in
                            (behavior_path, eyetracking_path)]
        self._video_meta = []
        for path in (behavior_meta_path, eye_meta_path):
            h5file = h5py.File(path, 'r')
            self._video_meta.append(eval(h5file['video_metadata'].value))
            h5file.close()

    def load_twop(self,
                  path=""):
        """
        Attempts to load a twop image sequence.
        """
        #self._twop_data = path or self._find_path("_output.avi")
        
    def load_platform(self,
                      path=""):
        """
        Attempts to load the platform metadata.
        """
        path = path or self._find_path("_platform.json")
        with open(path, 'r') as f:
            self._platform_data = json.load(f)
                            
    def load_auto(self):
        """
        Attempts to find and load data files if possible.
        """
        for f in [
            self.load_sync,
            self.load_pkl,
            self.load_videos,
            self.load_twop,
            self.load_platform,
        ]:
            try:
                f()
            except Exception:
                tb = traceback.format_exc()
                print("Call to {} failed: {}".format(f, tb))
        
    @property
    def timestamp(self):
        """
        Timestamp of the acquisition.  Is there a better one somewhere?
        """
        return self._platform_data['registration']['surface_2p']['acquired_at']
        
    @property
    def rig_id(self):
        """
        Rig ID that the data was acquired on.
        """
        return self._platform_data.get('rig_id', "NO PLATFORM FILE")
        
    @property
    def duration_info(self):
        """
        Gives the length of various data streams.
        """
        return {
            'stimulus': self.sync_data.stimulus_duration,
            'twop': self.sync_data.twop_duration,
            'video_monitoring': self.sync_data.video_duration,
        }
        
    def check_sync_signals(self):
        """
        Checks to ensure all required signals are present.

        TODO: make the required signals configurable.

        """
        print("Checking sync signals...")
        required = ["2p_vsync", "2p_trigger", "stim_vsync", "stim_photodiode",
                    "cam1_exposure", "cam2_exposure"]
        failure = False
        for signal in required:
            if not self.sync_data.signal_exists(signal):
                print(" - No {} events detected!")
                failure = True
        if not failure:
            print(" - All required signals accounted for!")
            return True
        else:
            return False

    @property
    def stim_vsyncs_pkl(self):
        """
        Stimulus vsyncs recorded in pkl file.
        """
        return self._pkl_data['vsynccount']
        
    @property
    def stim_vsyncs_sync(self):
        """
        Stimulus vsyncs detected by sync.
        """
        return len(self.sync_data.get_stim_vsyncs())
        
    @property
    def stim_script(self):
        """
        The stimulus script.
        
        ##TODO: this is a really shitty way to get the script name.  it just
            fetches it from the doc string that is at the top of all the
            production scripts.  We need a better solution than this.
        """
        for line in self._pkl_data['scripttext'].splitlines():
            if ".py" in line:
                return line
        else:
            return self._pkl_data['script']

    def check_stim_vsyncs(self):
        """
        Checks to ensure that there are the correct # of stim vsyncs.
        """
        print("Checking stim vsyncs...")
        sync_vsync_count = self.stim_vsyncs_sync
        pkl_vsync_count = self.stim_vsyncs_pkl
        
        if sync_vsync_count == pkl_vsync_count:
            print(" - Stim vsyncs accounted for: {}".format(sync_vsync_count))
            return True
        else:
            print(" - Stim vsync mismatch: {} in pkl but {} in sync".format(
                pkl_vsync_count, sync_vsync_count))
            return False

    def check_twop_vsyncs(self):
        """
        Checks to ensure that there are the correct # of twop vsyncs.
        """

    @property
    def photodiode_events_pkl(self):
        """
        The number of photodiode transitions we expect to have in our stimulus.
        """
        stim_vsyncs = self.stim_vsyncs_pkl
        photodiode_half_period = self._pkl_data['config']['syncsqrfreq']
        return stim_vsyncs / photodiode_half_period + 1  # add one for initial edge
        
    @property
    def photodiode_events_sync(self):
        """
        Number of photodiode transitions detected by sync.
        """
        return len(self.sync_data.get_photodiode_events())
        
    @property
    def photodiode_anomalies(self):
        """
        Number of anomalies in photodiode signal.
        """
        return len(self.sync_data.get_photodiode_anomalies()['indices'][0])

    def check_stim_photodiode(self):
        """
        Checks photodiode data to ensure that the number of events in sync
            matches the number that should have been displayed.
        """
        success = True        
        
        print("Checking photodiode events...")
        ideal_photodiode_events = self.photodiode_events_pkl
        sync_photodiode_events = self.photodiode_events_sync
        anomalies = self.photodiode_anomalies
        
        if sync_photodiode_events - anomalies == ideal_photodiode_events:
            print(" - Photodiode sync events match pkl file: {} events with {} anomalies.".format(
                ideal_photodiode_events, anomalies))
        else:
            print(" - Photodiode sync events do not match expected.")
            success = False
            
        display_lag = self.sync_data.display_lag
        if 0.1 > display_lag > 0:
            print(" - Display lag within expected range (0-0.1): {}".format(display_lag))
        else:
            print(" - Display lag not within expected range (0-0.1): {}".format(display_lag))
            success = False
        return success
        
    @property
    def video_frames_meta(self):
        """
        Number of frames recorded in the video metadata.
        """
        return [metadata['frames'] for metadata
                in self._video_meta]
                
    @property
    def video_frames_avi(self):
        """
        Number of frames in the actual AVI file.
        """
        return [int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) for video in
                self._video_data]
                
    @property
    def video_frames_sync(self):
        """
        Number of video monitoring frames recorded in the sync file.
        """
        return [len(f) for f in self.sync_data.get_video_vsyncs()]

    @property
    def video_vsyncs(self):
        """
        Returns the time points for the video monitoring vsyncs.
        """
        return self.sync_data.get_video_vsyncs()
            
    def check_video_frames(self):
        """
        Checks to ensure that the number of video frames match the sync and
            video metadata file.
        """
        success = True        
        
        print("Checking video metadata frame match...")
        for frames_in_video, frames_in_metadata, frames_in_sync in zip(
            self.video_frames_avi, self.video_frames_meta, self.video_frames_sync):
            if frames_in_video == frames_in_metadata:
                print(" - Frame count in video match metadata: {}".format(
                      frames_in_video))
            else:
                print(" - Frame count in video mismatch: {} in video {} in metadata.".format(
                      frames_in_video, frames_in_metadata))
                success = False

            if frames_in_video == frames_in_sync:
                print(" - Frame count in video match sync: {}".format(
                      frames_in_video))
            else:
                
                print(" - Frame count in video mismatch: {} in video, {} in sync.".format(
                      frames_in_video, frames_in_sync))
                success = False
        return success
        
    @property
    def encoder_data(self):
        """
        The encoder data recorded by the visual stim program.
        """
        try:
            return self._pkl_data['items']['foraging']['encoders'][0]['dx']
        except KeyError:
            # old pkl format
            return self._pkl_data['dx']
            
    @property
    def distance_travelled(self):
        """
        Total distance that the mouse travelled.
        """
        total_degrees = np.sum(self.encoder_data)
        total_radians = 0.0174533 * total_degrees
        mouse_radius = 8.89*2/3 # how far from the center of the wheel does the mouse run
        return total_radians * mouse_radius
                
    def check_encoder_data(self):
        """
        Checks for encoder data.
        """
        print("Checking for encoder data...")
        # TODO: check to ensure values are realistic.
        if len(self.encoder_data) > 0:
            print(" - Encoder data found.")
            return True
        else:
            print(" - No encoder data found.")
            return False
            
    def plot_encoder_data(self, out_file=""):
        """
        Plots the visual stimulus frame histogram.  This is good for
            visualizing how "clean" your frame intervals are.
        """
        encoder_data = self.encoder_data * 0.0174533 * 8.89 * 2 / 3 * 60
        plt.plot(encoder_data)
        plt.ylabel("cm/sec")
        plt.xlabel("frame number")
        fig = plt.gcf()
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return fig
            
    def check_all(self):
        self.check_encoder_data()
        self.check_stim_photodiode()
        self.check_stim_vsyncs()
        self.check_sync_signals()
        self.check_twop_vsyncs()
        self.check_video_frames()
        
    def generate_pdf_report(self, path=""):
        from report import OphysReport
        rep = OphysReport(self)
        rep.to_pdf(path)
        
    def close(self):
        """
        Close all open files.
        """
        for meta in self._video_meta:
            meta.close()
        for video in self._video_data:
            video.release()
        # anyting else to close?


class TwoPMovie(object):
    """ A 2p image sequence. Unused until I figure out how to read nd2 files. """
    def __init__(self, path):
        super(TwoPMovie, self).__init__()
        self.path = path

        self._shape = (0, 0, 0)
        self._duration = 0.0

    @property
    def frame_count(self):
        return 0

    @property
    def duration(self):
        return self._duration
    
    @property
    def shape(self):
        return self._shape
    


if __name__ == "__main__":

    path = r"C:\Users\derricw\Desktop\Derric_Sync\527381033_241350_20160705\527381033_241350_20160705_sync.h5"
    dset = Dataset2p(path)
    