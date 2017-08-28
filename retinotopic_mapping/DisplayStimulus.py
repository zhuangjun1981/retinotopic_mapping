# -*- coding: utf-8 -*-
"""
Visual Stimulus codebase implements several classes to display stimulus routines.
Can display frame by frame or compress data for certain stimulus routines and
display by index. Used to manage information between experimental devices and
interact with `StimulusRoutines` to produce visual display and log data. May also
be used to save and export movies of experimental stimulus routines for
presentation.
"""
from psychopy import visual, event
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time


from tools import FileTools as ft
from tools.IO import nidaq as iodaq

def analyze_frames(ts, refresh_rate, check_point=(0.02, 0.033, 0.05, 0.1)):
    """
    Analyze frame durations of time stamp data.

    Computes relevant statistics with respect to the presentation
    of a given stimulus. The statistics are computed in order
    to understand the timing of the frames since the monitor refresh
    rate isn't always an accurate tool for timing.

    Parameters
    ----------
    ts : ndarray
        list of time stamps of each frame measured (in seconds).
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

    frame_duration = ts[1::] - ts[0:-1]
    plt.figure()
    plt.hist(frame_duration, bins=np.linspace(0.0, 0.05, num=51))
    refresh_rate = float(refresh_rate)

    num_frames = len(ts)
    disp_true = ts[-1]-ts[0]
    disp_expect = (len(ts)-1)/refresh_rate
    avg_frame_time = np.mean(frame_duration)*1000
    sdev_frame_time = np.std(frame_duration)*1000
    short_frame = min(frame_duration)*1000
    short_frame_ind = np.nonzero(frame_duration==np.min(frame_duration))[0][0]
    long_frame = max(frame_duration)*1000
    long_frame_ind = np.nonzero(frame_duration==np.max(frame_duration))[0][0]

    frame_stats = '\n'
    frame_stats += 'Total number of frames    : %d. \n' % num_frames
    frame_stats += 'Total length of display   : %.5f second. \n' % disp_true
    frame_stats += 'Expected length of display: %.5f second. \n' % disp_expect
    frame_stats += 'Mean of frame durations   : %.2f ms. \n' % avg_frame_time
    frame_stats += 'Standard deviation of frame : %.2f ms.\n' % sdev_frame_time
    frame_stats += 'Shortest frame: %.2f ms, index: %d. \n' % (short_frame,
                                                               short_frame_ind)
    frame_stats += 'Longest frame : %.2f ms, index: %d. \n' % (long_frame,
                                                               long_frame_ind)

    for i in range(len(check_point)):
        check_number = check_point[i]
        frame_number = len(frame_duration[frame_duration>check_number])
        frame_stats += 'Number of frames longer than %d ms: %d; %.2f%% \n' \
                       % (round(check_number*1000),
                          frame_number,
                          round(frame_number*10000/(len(ts)-1))/100)

    print frame_stats

    return frame_duration, frame_stats


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
                 display_iter=1,
                 display_order=1,
                 mouse_id='Test',
                 user_id='Name',
                 psychopy_mon='testMonitor',
                 is_interpolate=False,
                 is_triggered=False,
                 by_index=False,
                 trigger_NI_dev='Dev1',
                 trigger_NI_port=1,
                 trigger_NI_line=0,
                 is_sync_pulse=True,
                 sync_pulse_NI_dev='Dev1',
                 sync_pulse_NI_port=1,
                 sync_pulse_NI_line=1,
                 display_trigger_event="negative_edge",
                 display_screen=0,
                 initial_background_color=0,
                 file_num_NI_dev='Dev1',
                 file_num_NI_port='0',
                 file_num_NI_lines='0:7'):
        """
        initialize `DisplaySequence` object

        Parameters
        ----------
        log_dir : str
            system directory path to where log display will be saved.
        backupdir : str, optional
            copy of directory path to save backup, defaults to `None`.
        display_iter : int, optional
            defaults to `1`
        display_order : int, optional
            determines whether the stimulus is presented forward or backwards.
            If `1`, stimulus is presented forward, whereas if `-1`, stimulus is
            presented backwards. Defaults to `1`.
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
        by_index : bool, optional
            determines if stimulus is displayed by index which saves memory
            and should speed up routines. Note that not every stimulus can be
            displayed by index and hence the default value is `False`.
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
        display_trigger_event :
            should be one of "negative_edge", "positive_edge", "high_level",
            or "low_level". defaults to "negative_edge".
        display_screen :
            determines which monitor to display stimulus on. defaults to `0`.
        initial_background_color :
            defaults to `0`.
        file_num_NI_dev :
            defaults to 'Dev1',
        file_num_NI_port :
            defaults to `0`,
        file_num_NI_lines :
            defaults to '0:7'.
        """

        self.sequence = None
        self.seq_log = {}
        self.psychopy_mon = psychopy_mon
        self.is_interpolate = is_interpolate
        self.is_triggered = is_triggered
        self.by_index = by_index
        self.trigger_NI_dev = trigger_NI_dev
        self.trigger_NI_port = trigger_NI_port
        self.trigger_NI_line = trigger_NI_line
        self.display_trigger_event = display_trigger_event
        self.is_sync_pulse = is_sync_pulse
        self.sync_pulse_NI_dev = sync_pulse_NI_dev
        self.sync_pulse_NI_port = sync_pulse_NI_port
        self.sync_pulse_NI_line = sync_pulse_NI_line
        self.display_screen = display_screen
        self.initial_background_color = initial_background_color
        self.keep_display = None
        self.file_num_NI_dev = file_num_NI_dev
        self.file_num_NI_port = file_num_NI_port
        self.file_num_NI_lines = file_num_NI_lines

        if display_iter % 1 == 0:
            self.display_iter = display_iter
        else:
            raise ArithmeticError, "`display_iter` should be a whole number."

        self.display_order = display_order
        self.log_dir = log_dir
        self.backupdir = backupdir
        self.mouse_id = mouse_id
        self.user_id = user_id
        self.seq_log = None

        self.clear()



    def set_any_array(self, any_array, log_dict = None):
        """
        to display any numpy 3-d array.
        """
        if len(any_array.shape) != 3:
            raise LookupError, "Input numpy array should have dimension of 3!"

        vmax = np.amax(any_array).astype(np.float32)
        vmin = np.amin(any_array).astype(np.float32)
        v_range = (vmax-vmin)
        any_array_nor = ((any_array-vmin)/v_range).astype(np.float16)
        self.sequence = 2*(any_array_nor-0.5)

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
        if self.by_index:
            self.sequence, self.seq_log = stim.generate_movie_by_index()
            self.clear()

        else:
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
             resolution = (800,600)

        # test monitor refresh rate
        try:
            refresh_rate = self.seq_log['monitor']['refresh_rate']
        except KeyError:
            print "No monitor refresh rate information, assuming 60Hz.\n"
            refresh_rate = 60.

        #prepare display frames log
        if self.sequence is None:
            raise LookupError, "Please set the sequence to be displayed!!\n"
        try:
            seq_frames = self.seq_log['stimulation']['frames']
            if self.display_order == -1:
                 seq_frames = seq_frames[::-1]
            # generate display Frames
            self.display_frames=[]
            for i in range(self.display_iter):
                self.display_frames += seq_frames
        except Exception as e:
            print '{}: {}'.format(type(e), str(e))
            print "No frame information in seq_log dictionary."
            print "Setting display_frames to 'None'.\n"
            self.display_frames = None

        # calculate expected display time
        if self.by_index:
            index_to_display = self.seq_log['stimulation']['index_to_display']
            display_time = (float(len(index_to_display))
                               * self.display_iter/ refresh_rate)
        else:
            display_time = (float(self.sequence.shape[0]) *
                            self.display_iter / refresh_rate)
        print '\n Expected display time: ', display_time, ' seconds\n'

        # generate file name
        self._get_file_name()
        print 'File name:', self.file_name + '\n'


        # -----------------setup psychopy window and stimulus--------------
        # start psychopy window
        window = visual.Window(size=resolution,
                               monitor=self.psychopy_mon,
                               fullscr=True,
                               screen=self.display_screen,
                               color=self.initial_background_color)
        stim = visual.ImageStim(window, size=(2,2),
                                interpolate=self.is_interpolate)

        # initialize keep_display
        self.keep_display = True

        # handle display trigger
        if self.is_triggered:
            display_wait = self._wait_for_trigger(event=self.display_trigger_event)
            if not display_wait:
                window.close()
                self.clear()
                return None
            else:
                time.sleep(5.) # wait remote object to start

        # display sequence either frame by frame or by index
        if self.by_index:
            # display by index
            self._display_by_index(window, stim)
        else:
            # display frame by frame
            self._display(window, stim)

        self.save_log()

        #analyze frames
        try:
             self.frame_duration, self.frame_stats = \
             analyze_frames(ts=self.time_stamp,
                            refresh_rate = self.seq_log['monitor']['refresh_rate'])
        except KeyError:
            print "No monitor refresh rate information, assuming 60Hz."
            self.frame_duration, self.frame_stats = \
                 analyze_frames(ts = self.time_stamp, refresh_rate = 60.)

        #clear display data
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

        #check NI signal
        trigger_task = iodaq.DigitalInput(self.trigger_NI_dev,
                                         self.trigger_NI_port,
                                         self.trigger_NI_line)
        trigger_task.StartTask()

        print "Waiting for trigger: " + event + ' on ' + trigger_task.devstr

        if event == 'low_level':
            last_TTL = trigger_task.read()
            while last_TTL != 0 and self.keep_display:
                last_TTL = trigger_task.read()[0]
                self._update_display_status()
            else:
                if self.keep_display:
                     trigger_task.StopTask()
                     print 'Trigger detected. Start displaying...\n\n'
                     return True
                else:
                     trigger_task.StopTask()
                     print 'Manual stop signal detected. Stopping the program.'
                     return False
        elif event == 'high_level':
            last_TTL = trigger_task.read()[0]
            while last_TTL != 1 and self.keep_display:
                last_TTL = trigger_task.read()[0]
                self._update_display_status()
            else:
                if self.keep_display:
                     trigger_task.StopTask()
                     print 'Trigger detected. Start displaying...\n\n'
                     return True
                else:
                     trigger_task.StopTask()
                     print 'Manual stop signal detected. Stopping the program.'
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
                 print 'Manual stop signal detected. Stopping the program.'
                 return False
            trigger_task.StopTask()
            print 'Trigger detected. Start displaying...\n\n'
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
                 print 'Manual stop signal detected. Stopping the program.'
                 return False
            trigger_task.StopTask()
            print 'Trigger detected. Start displaying...\n\n'
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
                            '-' + \
                            self.seq_log['stimulation']['stim_name'] + \
                            '-M' + \
                            self.mouse_id + \
                            '-' + \
                            self.user_id
        except KeyError:
            self.file_name = datetime.datetime.now().strftime('%y%m%d%H%M%S') + \
                            '-' + 'customStim' + '-M' + self.mouse_id + '-' + \
                            self.user_id

        file_number = self._get_file_number()

        if self.is_triggered:
             self.file_name += '-' + str(file_number)+'-Triggered'
        else:
             self.file_name += '-' + str(file_number) + '-notTriggered'

    def _get_file_number(self):
        """
        get synced file number for log file name
        """

        try:
            file_num_task = iodaq.DigitalInput(self.file_num_NI_dev,
                                             self.file_num_NI_port,
                                             self.file_num_NI_lines)
            file_num_task.StartTask()
            array = file_num_task.read()
            num_str = (''.join([str(line) for line in array]))[::-1]
            file_number = int(num_str, 2)
            # print array, file_number
        except Exception as e:
            print e
            file_number = None

        return file_number

    def _display_by_index(self,window,stim):
        """ display by index routine for simpler stim routines """

        # display frames by index
        time_stamps = []
        start_time = time.clock()
        index_to_display = self.seq_log['stimulation']['index_to_display']
        num_iters = len(index_to_display)

        # print 'frame per iter:', num_iters

        if self.is_sync_pulse:
            syncPulseTask = iodaq.DigitalOutput(self.sync_pulse_NI_dev,
                                                self.sync_pulse_NI_port,
                                                self.sync_pulse_NI_line)
            syncPulseTask.StartTask()
            _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

        i = 0
        while self.keep_display and i < (num_iters*self.display_iter):

            if self.display_order == 1:
                # Then display sequence in order
                 frame_num = i % num_iters

            if self.display_order == -1:
                # Then display sequence backwards
                 frame_num = num_iters - (i % num_iters) -1

            frame_index = index_to_display[frame_num]

            # print 'i:', i, '; index_display_ind:', frame_num, '; frame_ind:', frame_index

            stim.setImage(self.sequence[frame_index][::-1])
            stim.draw()
            time_stamps.append(time.clock()-start_time)

            #set syncPuls signal
            if self.is_sync_pulse:
                 _ = syncPulseTask.write(np.array([1]).astype(np.uint8))

            #show visual stim
            window.flip()

            #set syncPuls signal
            if self.is_sync_pulse:
                _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

            self._update_display_status()
            i += 1

        stop_time = time.clock()
        window.close()

        if self.is_sync_pulse:
             syncPulseTask.StopTask()

        self.time_stamp = np.array(time_stamps)
        self.display_length = stop_time-start_time

        if self.display_frames is not None:
            self.display_frames = self.display_frames[:i]

        if self.keep_display == True:
             print '\nDisplay successfully completed.'

    def _display(self, window, stim):

        # display frames
        time_stamp=[]
        start_time = time.clock()
        singleRunFrames = self.sequence.shape[0]

        if self.is_sync_pulse:
            syncPulseTask = iodaq.DigitalOutput(self.sync_pulse_NI_dev,
                                                self.sync_pulse_NI_port,
                                                self.sync_pulse_NI_line)
            syncPulseTask.StartTask()
            _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

        i = 0

        while self.keep_display and i < (singleRunFrames * self.display_iter):

            if self.display_order == 1:
                # Then display sequence in order
                 frame_num = i % singleRunFrames

            if self.display_order == -1:
                # then display sequence backwards
                 frame_num = singleRunFrames - (i % singleRunFrames) -1

            stim.setImage(self.sequence[frame_num][::-1])
            stim.draw()
            time_stamp.append(time.clock()-start_time)

            #set syncPuls signal
            if self.is_sync_pulse:
                 _ = syncPulseTask.write(np.array([1]).astype(np.uint8))

            #show visual stim
            window.flip()
            #set syncPuls signal
            if self.is_sync_pulse:
                 _ = syncPulseTask.write(np.array([0]).astype(np.uint8))

            self._update_display_status()
            i=i+1

        stop_time = time.clock()
        window.close()

        if self.is_sync_pulse:
             syncPulseTask.StopTask()

        self.time_stamp = np.array(time_stamp)
        self.display_length = stop_time-start_time

        if self.display_frames is not None:
            self.display_frames = self.display_frames[:i]

        if self.keep_display == True:
             print '\nDisplay successfully completed.'

    def flag_to_close(self):
        self.keep_display = False

    def _update_display_status(self):

        if self.keep_display is None:
             raise LookupError, 'self.keep_display should start as True'

        #check keyboard input 'q' or 'escape'
        keyList = event.getKeys(['q','escape'])
        if len(keyList) > 0:
            self.keep_display = False
            print "Keyboard stop signal detected. Stop displaying. \n"

    def set_display_order(self, display_order):

        self.display_order = display_order
        self.clear()

    def set_display_iteration(self, display_iter):

        if display_iter % 1 == 0:
            self.display_iter = display_iter
        else:
            raise ArithmeticError, "`display_iter` should be a whole number."
        self.clear()

    def save_log(self):

        if self.display_length is None:
            self.clear()
            raise LookupError, "Please display sequence first!"

        if self.file_name is None:
            self._get_file_name()

        if self.keep_display == True:
            self.file_name += '-complete'
        elif self.keep_display == False:
            self.file_name += '-incomplete'

        #set up log object
        directory = self.log_dir + '\sequence_display_log'
        if not(os.path.isdir(directory)):
             os.makedirs(directory)

        logFile = dict(self.seq_log)
        displayLog = dict(self.__dict__)
        displayLog.pop('seq_log')
        displayLog.pop('sequence')
        logFile.update({'presentation':displayLog})

        file_name =  self.file_name + ".hdf5"

        #generate full log dictionary
        path = os.path.join(directory, file_name)
        ft.saveFile(path,logFile)
        print ".pkl file generated successfully."

        backupFileFolder = self._get_backup_folder()
        if backupFileFolder is not None:
            if not (os.path.isdir(backupFileFolder)):
                 os.makedirs(backupFileFolder)
            backupFilePath = os.path.join(backupFileFolder,file_name)
            ft.saveFile(backupFilePath,logFile)
            print ".pkl backup file generate successfully"
        else:
            print "did not find backup path, no backup has been saved."

    def _get_backup_folder(self):

        if self.file_name is None:
            raise LookupError, 'self.file_name not found.'
        else:

            if self.backupdir is not None:

                curr_date = self.file_name[0:6]
                stim_name = self.seq_log['stimulation']['stim_name']
                if 'KSstim' in stim_name:
                    backupFileFolder = \
                         os.path.join(self.backupdir,
                                      curr_date+'-M'+self.mouse_id+'-Retinotopy')
                else:
                    backupFileFolder = \
                         os.path.join(self.backupdir,
                                      curr_date+'-M'+self.mouse_id+'-'+stim_name)
                return backupFileFolder
            else:
                return None

    def clear(self):
        """ clear display information. """
        self.display_length = None
        self.time_stamp = None
        self.frame_duration = None
        self.display_frames = None
        self.frame_stats = None
        self.file_name = None
        self.keep_display = None

if __name__ == "__main__":
     pass