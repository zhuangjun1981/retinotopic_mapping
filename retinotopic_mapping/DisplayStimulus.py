'''
Visual Stimulus codebase implements several classes to display stimulus routines.
Can display frame by frame or compress data for certain stimulus routines and
display by index. Used to manage information between experimental devices and
interact with `StimulusRoutines` to produce visual display and log data. May also
be used to save and export movies of experimental stimulus routines for
presentation.
'''
from psychopy import visual, event
import PIL
import os
import datetime
import skimage.external.tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tools import FileTools as ft
from tools.IO import nidaq as iodaq


def analyze_frames(ts_start, ts_end, refresh_rate, check_point=(0.02, 0.033, 0.05, 0.1)):
    """
    Analyze frame durations of time stamp data.

    Computes relevant statistics with respect to the presentation
    of a given stimulus. The statistics are computed in order
    to understand the timing of the frames since the monitor refresh
    rate isn't always an accurate tool for timing.

    Parameters
    ----------
    ts_start : 1d array
        list of time stamps of each frame start (in seconds).
    ts_end: 1d array
        list of time stamps of each frame end (in seconds).
    refresh_rate : float
        the refresh rate of imaging monitor measured (in Hz).
    check_point : tuple, optional

    Returns
    -------
    frame_duration : ndarray
        list containing the length of each time stamp.
    frame_stats : str
        string containing a statistical analysis of the image frames.

    """

    frame_interval = np.diff(ts_start)
    plt.figure()
    plt.hist(frame_interval, bins=np.linspace(0.0, 0.05, num=51))
    refresh_rate = float(refresh_rate)

    num_frames = ts_start.shape[0]
    disp_true = ts_end[-1] - ts_start[0]
    disp_expect = float(num_frames) / refresh_rate
    avg_frame_time = np.mean(frame_interval) * 1000
    sdev_frame_time = np.std(frame_interval) * 1000
    short_frame = min(frame_interval) * 1000
    short_frame_ind = np.where(frame_interval == np.min(frame_interval))[0][0]
    long_frame = max(frame_interval) * 1000
    long_frame_ind = np.where(frame_interval == np.max(frame_interval))[0][0]

    frame_stats = ''
    frame_stats += '\nTotal number of frames      : {}.'.format(num_frames)
    frame_stats += '\nTotal length of display     : {:.5f} second.'.format(disp_true)
    frame_stats += '\nExpected length of display  : {:.5f} second.'.format(disp_expect)
    frame_stats += '\nMean of frame intervals     : {:.2f} ms.'.format(avg_frame_time)
    frame_stats += '\nS.D. of frame intervals     : {:.2f} ms.'.format(sdev_frame_time)
    frame_stats += '\nShortest frame: {:.2f} ms, index: {}.'.format(short_frame, short_frame_ind)
    frame_stats += '\nLongest frame : {:.2f} ms, index: {}.'.format(long_frame, long_frame_ind)

    for i in range(len(check_point)):
        check_number = check_point[i]
        frame_number = len(frame_interval[frame_interval > check_number])
        frame_stats += '\nNumber of frames longer than {:5.3f} second: {}; {:6.2f}%'. \
            format(check_number, frame_number, (float(frame_number) * 100 / num_frames))

    print(frame_stats)

    return frame_interval, frame_stats


class DisplaySequence(object):
    """
    Display the stimulus routine from memory.

    Takes care of high level management of your computer
    hardware with respect to its interactions within a given experiment.
    Stimulus presentation routines are specified and external connection
    to National Instuments hardware devices is provided. Also takes care
    of the logging of relevant experimental data collected and where it
    will be stored on the computer used for the experiment.


    """

    def __init__(self,
                 log_dir,
                 backupdir=None,
                 identifier='000',
                 display_iter=1,
                 mouse_id='Test',
                 user_id='Name',
                 psychopy_mon='testMonitor',
                 is_by_index=True,
                 is_interpolate=False,
                 is_triggered=False,
                 is_save_sequence=False,
                 trigger_event="negative_edge",
                 trigger_NI_dev='Dev1',
                 trigger_NI_port=1,
                 trigger_NI_line=0,
                 is_sync_pulse=True,
                 sync_pulse_NI_dev='Dev1',
                 sync_pulse_NI_port=1,
                 sync_pulse_NI_line=1,
                 display_screen=0,
                 initial_background_color=0.,
                 color_weights=(1., 1., 1.)):
        """
        initialize `DisplaySequence` object

        Parameters
        ----------
        log_dir : str
            system directory path to where log display will be saved.
        backupdir : str, optional
            copy of directory path to save backup, defaults to `None`.
        identifier: str, optional
            identifing string for this particular experiment, this will
            show up in the name of log file when display is done.
        display_iter : int, optional
            defaults to `1`
        mouse_id : str, optional
            label for mouse, defaults to 'Test'.
        user_id : str, optional
            label for person performing experiment, defaults to 'Name'.
        psychopy_mon : str, optional
            label for monitor used for displaying the stimulus, defaults to
            'testMonitor'.
        is_interpolate : bool, optional
            defaults to `False`.
        is_triggered : bool, optional
            if `True`, stimulus will not display until triggered. if `False`,
            stimulus will display automatically. defaults to `False`.
        is_by_index : bool, optional
            determines if stimulus is displayed by index which saves memory
            and should speed up routines. Note that not every stimulus can be
            displayed by index and hence the default value is `False`.
        is_save_sequence : bool, optional
            defaults to False
            if True, the class will save the sequence of images to be displayed
            as a tif file, in the same folder of log file. If self.is_by_index
            is True, only unique frames will be saved. Note, this will save
            the whole sequence even if the display is interrupted in the middle.
        trigger_NI_dev : str, optional
            defaults to 'Dev1'.
        trigger_NI_port : int, optional
            defaults to `1`.
        trigger_NI_line : int, optional
            defaults to `0`.
        is_sync_pulse : bool, optional
            defaults to `True`.
        sync_pulse_NI_dev : str, optional
            defaults to 'Dev1'.
        sync_pulse_NI_port : int, optional
            defaults to 1.
        sync_pulse_NI_line : int, optional
            defaults to 1.
        trigger_event : str
            should be one of "negative_edge", "positive_edge", "high_level",
            or "low_level". defaults to "negative_edge".
        display_screen : int
            determines which monitor to display stimulus on. defaults to `0`.
        initial_background_color : float
            defaults to `0`. should be in the range from -1. (black) to 1. (white)
        color_weights : tuple, optional
            defaults to (1., 1., 1.)
            This should be a tuple with 3 elements. Each element specifies the
            weight of each color channel (R, G, B). The value range of each
            element is in the range [0., 1.]. This is designed in such way that
            if you want to suppress a certain channel i.e. red channel, you can
            change this parameter to (0., 1., 1.)
        """

        self.sequence = None
        self.seq_log = {}
        self.identifier = str(identifier)
        self.psychopy_mon = psychopy_mon
        self.is_interpolate = is_interpolate
        self.is_triggered = is_triggered
        self.is_by_index = is_by_index
        self.is_save_sequence = is_save_sequence
        self.trigger_NI_dev = trigger_NI_dev
        self.trigger_NI_port = trigger_NI_port
        self.trigger_NI_line = trigger_NI_line
        self.trigger_event = trigger_event
        self.is_sync_pulse = is_sync_pulse
        self.sync_pulse_NI_dev = sync_pulse_NI_dev
        self.sync_pulse_NI_port = sync_pulse_NI_port
        self.sync_pulse_NI_line = sync_pulse_NI_line
        self.display_screen = display_screen
        self.initial_background_color = float(initial_background_color)

        if len(color_weights) != 3:
            raise ValueError('input color_weights should be a tuple with 3 elements.')
        for cw in color_weights:
            if cw < -1. or cw > 1.:
                raise ValueError('each element of color_weight should be no less than -1. and no greater than 1.')
        self.color_weights = color_weights

        self.keep_display = None

        if display_iter % 1 == 0:
            self.display_iter = display_iter
        else:
            raise ArithmeticError, "`display_iter` should be a whole number."

        self.log_dir = log_dir
        self.backupdir = backupdir
        self.mouse_id = mouse_id
        self.user_id = user_id
        self.seq_log = None

        self.clear()

    def set_any_array(self, any_array, log_dict=None):
        """
        to display any numpy 3-d array.
        """
        if len(any_array.shape) != 3:
            raise LookupError, "Input numpy array should have dimension of 3!"

        vmax = np.amax(any_array).astype(np.float32)
        vmin = np.amin(any_array).astype(np.float32)
        v_range = (vmax - vmin)
        any_array_nor = ((any_array - vmin) / v_range).astype(np.float16)
        self.sequence = 2 * (any_array_nor - 0.5)

        if log_dict != None:
            if type(log_dict) is dict:
                self.seq_log = log_dict
            else:
                raise ValueError, '`log_dict` should be a dictionary!'
        else:
            self.seq_log = {}
        self.clear()

    def set_stim(self, stim):
        """
        Calls the `generate_movie` method of the respective stim object and
        populates the attributes `self.sequence` and `self.seq_log`

        Parameters
        ----------
        stim : Stim object
            the type of stimulus to be presented in the experiment
        """
        if self.is_by_index:
            if stim.stim_name in ['KSstim', 'KSstimAllDir']:
                raise LookupError('Stimulus {} does not support indexed display.'.format(stim.name))

            self.sequence, self.seq_log = stim.generate_movie_by_index()
            self.clear()

        else:
            if stim.stim_name in ['LocallySparseNoise', 'StaticGratingCircle', 'NaturalScene']:
                raise LookupError('Stimulus {} does not support full sequence display. Please use '
                                  'indexed display instead (set self.is_by_index = True).')

            self.sequence, self.seq_log = stim.generate_movie()
            self.clear()

    def trigger_display(self):
        """
        Display stimulus, initialize and perform global experimental routines.

        Prepares all of the necessary parameters to display stimulus and store
        the data collected in the experiment. Interacts with PyschoPy to create
        and display each frame of the selected stimulus routine. Handles
        global calls to trigger and timing devices within the experimental setup.

        Examples
        --------
        >>> # Assume monitor, indicator, and stimulus objects are defined
        >>> import DisplaySequence
        >>> ds = DisplaySequence(log_dir=r'C:\)
        >>> ds.set_stim(uniform_contrast)
        >>> ds.trigger_display()

        """
        # --------------- early preparation for display--------------------
        # test monitor resolution
        try:
            resolution = self.seq_log['monitor']['resolution'][::-1]
        except KeyError:
            resolution = (800, 600)

        # test monitor refresh rate
        try:
            refresh_rate = self.seq_log['monitor']['refresh_rate']
        except KeyError:
            print("No monitor refresh rate information, assuming 60Hz.\n")
            refresh_rate = 60.

        # prepare display frames log
        if self.sequence is None:
            raise LookupError("Please set the sequence to be displayed by using self.set_stim().\n")
        if not self.seq_log:
            raise LookupError("Please set the sequence log dictionary to be displayed "
                              "by using self.set_stim().\n")

        # if display by index, check frame indices were not larger than the number of frames in
        # self.sequence
        if self.is_by_index:
            max_index = max(self.seq_log['stimulation']['index_to_display'])
            min_index = min(self.seq_log['stimulation']['index_to_display'])
            if max_index >= self.sequence.shape[0] or min_index < 0:
                raise ValueError('Max display index range: {} is out of self.sequence frame range: {}.'
                                 .format((min_index, max_index), (0, self.sequence.shape[0] - 1)))
            if 'frames_unique' not in self.seq_log['stimulation'].keys():
                raise LookupError('"frames_unique" is not found in self.seq_log["stimulation"]. This'
                                  'is required when display by index.')
        else:
            if 'frames' not in self.seq_log['stimulation'].keys():
                raise LookupError('"frames" is not found in self.seq_log["stimulation"]. This'
                                  'is required when display by full sequence.')

        # calculate expected display time
        if self.is_by_index:
            index_to_display = self.seq_log['stimulation']['index_to_display']
            display_time = (float(len(index_to_display)) *
                            self.display_iter / refresh_rate)
        else:
            display_time = (float(self.sequence.shape[0]) *
                            self.display_iter / refresh_rate)
        print('\nExpected display time: {} seconds.\n'.format(display_time))

        # generate file name
        self._get_file_name()
        print('File name: {}.\n'.format(self.file_name))

        # -----------------setup psychopy window and stimulus--------------
        # start psychopy window
        window = visual.Window(size=resolution,
                               monitor=self.psychopy_mon,
                               fullscr=True,
                               screen=self.display_screen,
                               color=self.initial_background_color)

        stim = visual.ImageStim(window, size=(2, 2), interpolate=self.is_interpolate)

        # initialize keep_display
        self.keep_display = True

        # handle display trigger
        if self.is_triggered:
            display_wait = self._wait_for_trigger(event=self.trigger_event)
            if not display_wait:
                window.close()
                self.clear()
                return None
            else:
                time.sleep(5.)  # wait remote object to start

        # actual display
        self._display(window=window, stim=stim)

        # analyze frames
        try:
            self.frame_duration, self.frame_stats = \
                analyze_frames(ts_start=self.frame_ts_start, ts_end=self.frame_ts_end,
                               refresh_rate=self.seq_log['monitor']['refresh_rate'])
        except KeyError:
            print("No monitor refresh rate information, assuming 60Hz.")
            self.frame_duration, self.frame_stats = \
                analyze_frames(ts_start=self.frame_ts_start, ts_end=self.frame_ts_end, refresh_rate=60.)

        self.save_log()

        # clear display data
        self.clear()

    def _wait_for_trigger(self, event):
        """
        time place holder for waiting for trigger

        Parameters
        ----------
        event : str from {'low_level','high_level','negative_edge','positive_edge'}
            an event triggered via a National Instuments experimental device.
        Returns
        -------
        Bool :
            returns `True` if trigger is detected and `False` if manual stop
            signal is detected.
        """

        # check NI signal
        trigger_task = iodaq.DigitalInput(self.trigger_NI_dev,
                                          self.trigger_NI_port,
                                          self.trigger_NI_line)
        trigger_task.StartTask()

        print("Waiting for trigger: {} on {}.".format(event, trigger_task.devstr))

        if event == 'low_level':
            last_TTL = trigger_task.read()
            while last_TTL != 0 and self.keep_display:
                last_TTL = trigger_task.read()[0]
                self._update_display_status()
            else:
                if self.keep_display:
                    trigger_task.StopTask()
                    print('Trigger detected. Start displaying...\n\n')
                    return True
                else:
                    trigger_task.StopTask()
                    print('Keyboard interrupting signal detected. Stopping the program.')
                    return False
        elif event == 'high_level':
            last_TTL = trigger_task.read()[0]
            while last_TTL != 1 and self.keep_display:
                last_TTL = trigger_task.read()[0]
                self._update_display_status()
            else:
                if self.keep_display:
                    trigger_task.StopTask()
                    print('Trigger detected. Start displaying...\n\n')
                    return True
                else:
                    trigger_task.StopTask()
                    print('Keyboard interrupting signal detected. Stopping the program.')
                    return False
        elif event == 'negative_edge':
            last_TTL = trigger_task.read()[0]
            while self.keep_display:
                current_TTL = trigger_task.read()[0]
                if (last_TTL == 1) and (current_TTL == 0):
                    break
                else:
                    last_TTL = int(current_TTL)
                    self._update_display_status()
            else:
                trigger_task.StopTask()
                print('Keyboard interrupting signal detected. Stopping the program.')
                return False
            trigger_task.StopTask()
            print('Trigger detected. Start displaying...\n\n')
            return True
        elif event == 'positive_edge':
            last_TTL = trigger_task.read()[0]
            while self.keep_display:
                current_TTL = trigger_task.read()[0]
                if (last_TTL == 0) and (current_TTL == 1):
                    break
                else:
                    last_TTL = int(current_TTL)
                    self._update_display_status()
            else:
                trigger_task.StopTask();
                print('Keyboard interrupting signal detected. Stopping the program.')
                return False
            trigger_task.StopTask()
            print('Trigger detected. Start displaying...\n\n')
            return True
        else:
            raise NameError, "`trigger` not in " \
                             "{'negative_edge','positive_edge', 'high_level','low_level'}!"

    def _get_file_name(self):
        """
        generate the file name of log file
        """

        try:
            self.file_name = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                             '-' + self.seq_log['stimulation']['stim_name'] + \
                             '-M' + self.mouse_id + '-' + self.user_id + '-' + \
                             self.identifier
        except KeyError:
            self.file_name = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                             '-' + 'customStim' + '-M' + self.mouse_id + '-' + \
                             self.user_id + '-' + self.identifier

        if self.is_triggered:
            self.file_name += '-Triggered'
        else:
            self.file_name += '-notTriggered'

    def _display(self, window, stim):
        """
        display stimulus
        """
        frame_ts_start = []
        frame_ts_end = []
        start_time = time.clock()

        if self.is_by_index:
            index_to_display = self.seq_log['stimulation']['index_to_display']
            iter_frame_num = len(index_to_display)
        else:
            iter_frame_num = self.sequence.shape[0]
            index_to_display = range(iter_frame_num)

        # print('frame per iter: {}'.format(iter_frame_num))

        if self.is_sync_pulse:
            syncPulseTask = iodaq.DigitalOutput(self.sync_pulse_NI_dev,
                                                self.sync_pulse_NI_port,
                                                self.sync_pulse_NI_line)
            syncPulseTask.StartTask()
            _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

        i = 0
        self.displayed_frames = []

        while self.keep_display and i < (iter_frame_num * self.display_iter):

            frame_num = i % iter_frame_num
            frame_index = index_to_display[frame_num]

            # print('i:{}; index_display_ind:{}; frame_ind{}.'.format(i, frame_num, frame_index))

            if self.color_weights == (1., 1., 1.):
                stim.setImage(self.sequence[frame_index][::-1])
            else:
                curr_frame = self.sequence[frame_index]
                curr_frame = ((curr_frame + 1.) * 255 / 2.)
                curr_frame_r = PIL.Image.fromarray((curr_frame * self.color_weights[0]).astype(np.uint8))
                curr_frame_g = PIL.Image.fromarray((curr_frame * self.color_weights[1]).astype(np.uint8))
                curr_frame_b = PIL.Image.fromarray((curr_frame * self.color_weights[2]).astype(np.uint8))
                curr_frame = PIL.Image.merge('RGB', (curr_frame_r, curr_frame_g, curr_frame_b))
                # plt.imshow(curr_frame)
                # plt.show()
                stim.setImage(curr_frame)

            stim.draw()

            # set sync pulse start signal
            if self.is_sync_pulse:
                _ = syncPulseTask.write(np.array([1]).astype(np.uint8))

            # save frame start timestamp
            frame_ts_start.append(time.clock() - start_time)

            # show visual stim
            window.flip()

            # save displayed frame information
            if self.is_by_index:
                self.displayed_frames.append(self.seq_log['stimulation']['frames_unique'][frame_index])
            else:
                self.displayed_frames.append(self.seq_log['stimulation']['frames'][frame_index])

            # save frame end timestamp
            frame_ts_end.append(time.clock() - start_time)

            # set sync pulse end signal
            if self.is_sync_pulse:
                _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

            self._update_display_status()
            i += 1

        stop_time = time.clock()
        window.close()

        if self.is_sync_pulse:
            syncPulseTask.StopTask()

        self.frame_ts_start = np.array(frame_ts_start)
        self.frame_ts_end = np.array(frame_ts_end)
        self.display_length = stop_time - start_time

        if self.keep_display == True:
            print('\nDisplay successfully completed.')

    def flag_to_close(self):
        self.keep_display = False

    def _update_display_status(self):

        if self.keep_display is None:
            raise LookupError, 'self.keep_display should start as True'

        # check keyboard input 'q' or 'escape'
        keyList = event.getKeys(['q', 'escape'])
        if len(keyList) > 0:
            self.keep_display = False
            print("Keyboard interrupting signal detected. Stop displaying. \n")

    def set_display_iteration(self, display_iter):

        if display_iter % 1 == 0:
            self.display_iter = display_iter
        else:
            raise ArithmeticError, "`display_iter` should be a whole number."
        self.clear()

    def save_log(self):

        if self.display_length is None:
            self.clear()
            raise LookupError("Please display sequence first!")

        if self.file_name is None:
            self._get_file_name()

        if self.keep_display == True:
            self.file_name += '-complete'
        elif self.keep_display == False:
            self.file_name += '-incomplete'

        # set up log object
        directory = os.path.join(self.log_dir, 'visual_display_log')
        if not (os.path.isdir(directory)):
            os.makedirs(directory)

        logFile = dict(self.seq_log)
        displayLog = dict(self.__dict__)
        displayLog.pop('seq_log')
        displayLog.pop('sequence')
        logFile.update({'presentation': displayLog})

        file_name = self.file_name + ".pkl"

        # generate full log dictionary
        path = os.path.join(directory, file_name)
        ft.saveFile(path, logFile)
        if self.is_save_sequence:
            tf.imsave(os.path.join(directory, self.file_name + '.tif'),
                      self.sequence.astype(np.float32))
        print(".pkl file generated successfully.")

        backupFileFolder = self._get_backup_folder()
        if backupFileFolder is not None:
            if not (os.path.isdir(backupFileFolder)):
                os.makedirs(backupFileFolder)
            backupFilePath = os.path.join(backupFileFolder, file_name)
            ft.saveFile(backupFilePath, logFile)

            if self.is_save_sequence:
                tf.imsave(os.path.join(backupFileFolder, self.file_name + '.tif'),
                          self.sequence.astype(np.float32))
            print(".pkl backup file generate successfully")
        else:
            print("Did not find backup path, no backup has been saved.")

    def _get_backup_folder(self):
        if self.backupdir is not None:
            backup_folder = os.path.join(self.backupdir, 'visual_display_log')
            if not os.path.isdir(backup_folder):
                os.makedirs(backup_folder)
            return backup_folder
        else:
            return None

    def clear(self):
        """ clear display information. """
        self.display_length = None
        self.time_stamp = None
        self.frame_duration = None
        self.displayed_frames = None
        self.frame_stats = None
        self.file_name = None
        self.keep_display = None


if __name__ == "__main__":
    pass
