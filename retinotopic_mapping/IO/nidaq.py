'''
Created on Oct 26, 2012

@author: derricw

#------------------------------------------------------------------------------
nidaq.py
#------------------------------------------------------------------------------

Derric's wrapper for DAQ from an NIDAQ board using the PyDAQmx library.

Dependencies:
Python27
PyDAQmx (http://pypi.python.org/pypi/PyDAQmx) #Tested using 1.2.3
numpy (http://www.scipy.org/Download)
scipy (http://www.scipy.org/Download)

NIDAQmc C Reference:  #PyDAQmx maps one-to-one to the C library
http://zone.ni.com/reference/en-XX/help/370471W-01/

'''

import logging

from PyDAQmx import Task
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxFunctions import function_dict, function_list
import PyDAQmx.DAQmxFunctions as DAQmxFunctions
from numpy import zeros, sin, arange, pi, array, ones
import numpy as np
from ctypes import c_long, c_ulong, CFUNCTYPE, POINTER
from ctypes import create_string_buffer, c_double, c_void_p, c_char_p

from toolbox.misc.timer import timeit


##############################################################################
# System Object
##############################################################################

system_function_list = [name for name in function_dict.keys() if \
                        "DAQmxGetSys" in name]

def _create_system_method(func):
    """
    Creates a System class method from a NIDAQmx function.
    """
    def _call_method(self, *args):
        return func(*args)
    return _call_method

def _create_system_buffer_method(func):
    """
    Creates a System class method from a NIDAQmx function designed to
        parse a buffer.
    """
    def _call_method(self):
        buff = " "*self.buffer_size
        func(buff, self.buffer_size)
        data = buff.strip().strip("\x00").split(', ')
        if data[0]=='':
            data.remove('')
        return data
    return _call_method

class System(object):
    """
    System state tracking.

    Autopopulated with the PyDAQmx methods associated with the system state.

    Added convenience methods as well for pythonicness.

    Examples:
        >>> s = System()
        >>> s.getDevNames()
        ['Dev1', 'Dev2']

    """
    def __init__(self):
        super(System, self).__init__()
        self.buffer_size = 4096

    def _get_property_u32(self, method):
        data = c_ulong()
        method(data)
        return data

    def getNIDAQVersion(self):
        major = self._get_property_u32(self.GetSysNIDAQMajorVersion).value
        minor = self._get_property_u32(self.GetSysNIDAQMinorVersion).value
        update = self._get_property_u32(self.GetSysNIDAQUpdateVersion).value
        return "{}.{}.{}".format(major, minor, update)

# Here we add functions to the System class
#   Functions with a char buffer for the first object are properties whose
#   values are written to long buffers so we given them a helper function
#   so that the user doesn't have to deal with it.
for function_name in system_function_list:
    name = function_name[5:]
    func = getattr(DAQmxFunctions, function_name)

    arg_names = function_dict[function_name]['arg_name']
    arg_types = function_dict[function_name]['arg_type']

    if len(arg_types) > 0 and (arg_types[0] is c_char_p):
        system_func = _create_system_buffer_method(func)
        name = name.replace("GetSys", "get")
    else:
        system_func = _create_system_method(func)
    system_func.__name__ = name
    system_func.__doc__ = 'S.%s(%s) -> error.' % \
            (name, ', '.join(arg_names[1:]))
    setattr(System, name, system_func)

#clean namespace a bit
del _create_system_method
del system_function_list

##############################################################################
# Device Object
##############################################################################

device_func_list = [name for name in function_dict.keys() if \
                    len(function_dict[name]['arg_type']) > 0 and \
                    (function_dict[name]['arg_type'][0] is c_char_p) and \
                    'device' in function_dict[name]['arg_name'][0]]

def _create_device_method(func):
    """
    Creates a System class method from a NIDAQmx function.
    """
    def _call_method(self, *args):
        return func(self.device_name, *args)
    return _call_method

def _create_device_buffer_method(func):
    """
    Creates a Device class method from a NIDAQmx function designed to
        parse a buffer.
    """
    def _call_method(self):
        buff = " "*self.buffer_size
        func(self.device_name, buff, self.buffer_size)
        data = buff.strip().strip("\x00").split(', ')
        if data[0]=='':
            data.remove('')
        return data
    return _call_method

class Device(object):
    """
    Device object.

    Autopopulated with functions that use "deviceName" as their first argument.

    Some methods (those that start with a lower-case letter), have been
        replaced with a method that automatically builds and parses the buffer


    Args:
        device_name (str): The device name Ex: "Dev1"

    Example:
        >>> d = Device('Dev1')
        >>> d.getDOPorts()
        ['Dev1/port0', 'Dev1/port1']

    """
    def __init__(self, device_name):
        super(Device, self).__init__()
        self.device_name = device_name
        self.buffer_size = 4096

    def _get_property_buffer(self, method):
        buff = " "*self.buffer_size
        method(buff, self.buffer_size)
        return buff.strip().strip("\x00").split(', ')

    def getAIChannels(self):
        return self.getAIPhysicalChans()

    def getAOChannels(self):
        return self.getAOPhysicalChans()

    def getCOChannels(self):
        return self.getCOPhysicalChans()

    def getCIChannels(self):
        return self.getCIPhysicalChans()

    def reset(self):
        return self.ResetDevice()
        
for function_name in device_func_list:
    name = function_name[5:]
    func = getattr(DAQmxFunctions, function_name)
    arg_names = function_dict[function_name]['arg_name']
    arg_types = function_dict[function_name]['arg_type']

    if len(arg_types) == 3 and (arg_types[1] is c_char_p) and \
        (arg_types[2] is c_ulong):
        devfunc = _create_device_buffer_method(func)
        name = name.replace("GetDev", "get")
        name = name.replace("Get", "get")
    else:
        devfunc = _create_device_method(func)
    devfunc.__name__ = name
    devfunc.__doc__ = 'D.%s(%s) -> error.' % \
            (name, ', '.join(arg_names[1:]))
    setattr(Device, name, devfunc)

del _create_device_method
del device_func_list

##############################################################################
# Task Objects
##############################################################################

class BaseTask(Task):
    """
    Base class for NIDAQmx tasks.

    Base tasks aren't pre-configured for anything.  They have some convenience
        methods for clock and trigger configuration, but haven't set up any
        channels for IO yet.

    They can still use all of the methods of the PyDAQmx Task object.

    Example:
        >>> from PyDAQmx.DAQmxConstants import *
        >>> import numpy as np
        >>> bt = BaseTask()
        >>> bt.CreateDOChan('Dev1/port0/line0:4',
                           '',
                           DAQmx_Val_ChanForAllLines)
        >>> bt.start()
        >>> buf = np.array([0,1,0,1], dtype=np.uint8)
        >>> bt.WriteDigitalLines(1, 0, 10.0, DAQmx_Val_GroupByChannel, buf,
                                None, None)
        >>> bt.stop()
        >>> bt.clear()

    """
    def __init__(self):
        Task.__init__(self)  # old style class
        self.__registered = False  # data callback not registered

    def start(self):
        """
        Starts the task.
        """
        self.StartTask()

    def stop(self):
        """
        Stops the task.  It can be restarted.
        """
        self.StopTask()

    def clear(self):
        """
        Clears the task.  It cannot be restarted.
        """
        try:
            self.stop()
        except Exception as e:
            ##TODO: catch specific type
            print e
            pass
        self.ClearTask()

    def cfg_sample_clock(self,
                         rate,
                         source="",
                         edge='rising',
                         mode='continuous',
                         buffer_size=1000,
                         ):
        """
        Configures the sample clock.

        Args:
            rate (float): Sample rate in Hz
            source (Optional[str]): name of source terminal
            edge (Optional[str]): rising or falling edge for example "r"
            mode (Optional[str]): sample mode for example "continuous"
            buffer_size (Optional[int]): write buffer size

        Examples:
            >>> mytask.cfg_sample_clock("/Dev1/ai/SampleClock", 'f', 'c', 1000)

        """
        edge = get_edge_val(edge)
        mode = get_mode_val(mode)

        status = self.CfgSampClkTiming(source, rate, edge, mode, buffer_size)
        self.buffer_size = buffer_size
        self.clock_speed = rate
        logging.debug("Sample clock configured to ({}, {}, {}, {}, {})".format(rate,
            source, edge, mode, buffer_size))
        return status

    def cfg_dig_start_trigger(self,
                              source,
                              edge='rising',
                              ):
        """
        Configures the start trigger.

        Args:
            source (str): Start trigger source.
            edge (str): rising or falling edge

        Examples:
            >>> mytask.cfg_digital_start_trigger("/Dev1/ai/StartTrigger",'r')

        """
        edge = get_edge_val(edge)
        self.CfgDigEdgeStartTrig(source, edge)
        logging.debug("Start trigger configured to ({}, {})".format(source, edge))

    def set_timebase_divisor(self, divisor=1):
        """
        Supposed to set the divisor for the clock's timebase.

        Doesn't seem to work...

        #TODO: Call NI and ask them why this doesn't work.

        """
        if divisor == 1:
            self.ResetSampClkTimebaseDiv()
        elif divisor > 1:
            self.SetSampClkTimebaseDiv(divisor)
            print divisor
        else:
            raise ValueError("Divisor must be between 1 and 2^32")

    def get_clock_terminal(self):
        """
        Returns the terminal for the sample clock.

        Example output: "/Dev1/ai/SampleClock"
        """
        buffer_size = 1024
        lines = " "*buffer_size
        self.GetSampClkTerm(lines, buffer_size)
        return lines.strip().strip('\x00').split(', ')[0]

    def get_start_trigger_term(self):
        """
        Returns the terminal for start trigger.
        """
        buffer_size = 1024
        lines = " "*buffer_size
        self.GetStartTrigTerm(lines, buffer_size)
        return lines.strip().strip('\x00').split(', ')[0]

    def register_sample_callback(self,
                                 buffer_size,
                                 direction='input',
                                 synchronous=False):
        """
        Register a sample callback for a buffer of N samples.
        """
        direction = get_direction_val(direction)
        synchronous = get_synchronous_val(synchronous)
        self.AutoRegisterEveryNSamplesEvent(direction,
                                            buffer_size,
                                            synchronous)
        self.__registered = True
        logging.debug("Task sample callback registered for {} samples.".format(buffer_size))

    def unregister_sample_callback(self,
                                   direction='input',
                                   synchronous=False):
        """
        Unregister a sample callback.
        """
        direction = get_direction_val(direction)
        synchronous = get_synchronous_val(synchronous)
        if self.__registered:
            self.RegisterEveryNSamplesEvent(direction,
                                            self.buffer_size,
                                            synchronous,
                                            DAQmxEveryNSamplesEventCallbackPtr(0),
                                            None)
            self.__registered = False
            logging.debug("Task sample callback unregistered.")
        else:
            logging.debug("Task already unregistered.")

#-------------------------------------------------------------- Analog Tasks


class AnalogInput(BaseTask):
    '''
    Gets analog input from NIDAQ device.
        Tested using several buffer sizes and channels on a NI USB-6210.

    Parameters
    ----------

    device : 'Dev1'
        NIDAQ device id
    channels : [0]
        List of channels to read
    buffer_size : 500
        Integer size of buffer to read
    clock_speed : 10000.0
        Float sample clock speed
    terminal_config : "RSE"
        String for terminal type: "RSE","Diff"
    voltage_range : [-10.0,10.0]
        Float bounds for voltages
    timout : 10.0
        Float timeout for read
    tdms : None
        tdms file to write to.
    binary : None
        binary file to write to
    dtype : np.float64
        output data type

    Returns
    -------

    AnalogInput : Task
        Task object

    Examples
    --------

    >>> ai = AnalogInput('Dev1',channels=[0],buffer_size=500)
    >>> ai.start()
    >>> for x in range(10):
    ...     time.sleep(1) #collects some data
    ...     print ai.data #prints the current buffer
    >>> ai.clear()

    '''
    def __init__(self,
                 device='Dev1',
                 channels=[0],
                 buffer_size=500, 
                 clock_speed=10000.0,
                 terminal_config="RSE",
                 voltage_range=[-10.0, 10.0],
                 timeout=10.0,
                 binary=None,
                 dtype=np.float64,
                 custom_callback=None):

        BaseTask.__init__(self)

        #set up task properties
        self.buffer_size = buffer_size
        self.clock_speed = clock_speed
        self.channels = channels
        self.data = zeros((self.buffer_size,
            len(self.channels)), dtype=np.float64)  # data buffer
        self.dataArray = []
        self.binary = binary
        self.terminal_config = get_input_terminal_config(terminal_config)
        self.voltage_range = voltage_range
        self.timeout = timeout
        self.dtype = dtype
        if custom_callback:
            self.callback = custom_callback
        else:
            self.callback = self.default_callback
        self.buffercount = 0

        #create dev str for various channels
        self.devstr = ""
        if type(channels) is int:
            channels = [channels]
        for channel in channels:
            self.devstr += str(device) + "/ai" + str(channel) + ","
        self.devstr = self.devstr[:-1]

        self.CreateAIVoltageChan(self.devstr, "", self.terminal_config,
                                 self.voltage_range[0], self.voltage_range[1],
                                 DAQmx_Val_Volts, None)

        self.cfg_sample_clock(rate=self.clock_speed,
                              edge='rising',
                              mode='continuous',
                              buffer_size=self.buffer_size)

        if self.binary is not None:
            self.outFile = open(self.binary, 'wb')

    def cfg_sample_clock(self,
                         rate=10000.0,
                         source="",
                         edge='rising',
                         mode='continuous',
                         buffer_size=1000):
        """
        Custom version of the clock config function.  Needs to re-register
            the NSamples callback.
        """

        # first unregister the old buffer callback if it is registered
        self.unregister_sample_callback()

        # then set up the sample clock
        BaseTask.cfg_sample_clock(self,
                                  rate=rate,
                                  source=source,
                                  edge=edge,
                                  mode=mode,
                                  buffer_size=buffer_size)

        # set up a new data buffer
        self.data = zeros((buffer_size, len(self.channels)),
                          dtype=np.float64)

        # then register the buffer callback
        self.register_sample_callback(buffer_size)

    def EveryNCallback(self):
        """
        Callback for buffer read.  Occurs automatically when `self.buffer_size`
            samples are read.
        """
        try:
            read = int32()
            # read into the data buffer
            self.ReadAnalogF64(self.buffer_size, self.timeout, DAQmx_Val_Auto,
                self.data, (self.buffer_size*len(self.channels)), byref(read),
                None)
            if self.binary:
                self.outFile.write(self.data.astype(self.dtype).tostring())
            self.callback(self.data)
            self.buffercount += 1
        except Exception as e:
            print("Failed to read buffer #%i -> %s" % (self.buffercount, e))

    def read(self, samples=1):
        """
        Syncrhonous read.
        """
        read = int32()
        output_size = len(self.channels)*samples
        output_array = np.zeros((len(self.channels), samples), dtype=np.float64)
        self.ReadAnalogF64(samples, self.timeout, DAQmx_Val_GroupByScanNumber,
                           output_array, output_size, byref(read),
                           None)
        return output_array

    def clear(self):
        BaseTask.clear(self)
        if self.binary:
            self.outFile.flush()
            self.outFile.close()

    def default_callback(self, data):
        return


class AnalogOutput(BaseTask):
    '''
    Analog Output task.  
        Writes arrays of float64's to an analog output channel at a specified sample rate.
        Value remains until changed, even after task is cleared.

    Parameters
    ----------

    device : GetDevices()[0]
        String, NIDAQ device id
    channels : [0]
        List of intergers, channels to write to
    voltage_range : [0.0, 0.5]
        Float, bounds for voltages

    Returns
    -------

    AnalogOutput : Task
        Task object

    Examples
    --------

    >>> data = 9.95*sin(arange(1000, dtype=float64)*2*pi/1000) #create waveform of some sort
    >>> ao = AnalogOutput('Dev1',channels=[0])
    >>> ao.start()
    >>> ao.write(data) #will write samples only once at the clock speed of board
    >>> ao.clear()

    TODO
    ----
    TODO: I'm having some trouble configuring clock timing.  My NI USB-6211 appears to write at exactly 1600 samples/s.
        I can set this in the NI Measurement explorer for continuous samples, but for some reason not here.
        The data sheet says it should be capable of 250kS/s per channel...

        Also, I'm having trouble outputting to multiple channels at once.  I think that it is something to do with
        DAQmx_Val_GroupByChannel in the WriteAnalogF64 function.


    '''
    def __init__(self,
                 device='Dev1',
                 channels=[0],
                 voltage_range=[0.0, 5.0]):

        BaseTask.__init__(self)

        #create dev str for various channels
        self.channels = channels
        self.voltage_range = voltage_range
        devStr = ""
        if type(channels) is int:
            channels = [channels]
        for channel in channels:
            devStr += str(device) + "/ao" + str(channel) + ","
        devStr = devStr[:-1]

        self.CreateAOVoltageChan(devStr, "", self.voltage_range[0],
                                 self.voltage_range[1], DAQmx_Val_Volts, None)

        #self.CfgOutputBuffer(buffer_size)
        #self.CfgSampClkTiming("",sampleRate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,buffer_size) #can't get this to work at the moment

        self.AutoRegisterDoneEvent(0)

    def write(self, data):
        """
        Writes a numpy array of float64's to the analog output.
        """
        status = self.WriteAnalogF64(len(data)/len(self.channels), 0, -1,
                                     DAQmx_Val_GroupByChannel, data, None,
                                     None)
        return status

    def Write(self, data):
        """
        Deprecated.
        """
        return self.write(data)

    def DoneCallback(self, status):
        """
        Done callback.  Unregistered at this point.  Might just eliminate it.
        """
        return 0


class AnalogFunctionOutput(BaseTask):
    """
    Creates a wave output function.

    Parameters
    ----------
    device : GetDevices()[0]
        String, NIDAQ device id
    channels : [0]
        List of integers, channels to write to
    ftype : 'sin'
        'sin','saw', 'sqr' or 'custom'
    frequency : 1
        wave frequency Hz
    amplitude : 5
        wave amplitude volts
    offset : 0
        wave offset volts
    phase : 0
        wave initial phase (radians)
    sample_rate : 1000
        output sample rate
    buffer_size : 1000
        write buffer size
    voltage_range : [-10.0, 10.0]
        Float bounds for voltages
    custom_wave : None
        custom waveform (see ftype above) Must be numpy array of float64's
        if multiply channels are desired, the custom array must be appropriately shaped

    Returns
    -------
    AnalogFunctionOutput : Task
        Task object

    Examples
    --------

    >>> afo = AnalogFunctionOutput('Dev1',channels=[0],ftype='sin',frequency=1)
    >>> afo.start()
    >>> time.sleep(10) #output for 10 seconds
    >>> afo.clear()

    TODO
    ----

    Outputing two different waves simultaneously on two different channels.
    Duplicate waves on two different channels works fine.

    """
    def __init__(self,
                 device='Dev1',
                 channels=[0],
                 ftype='sin',
                 frequency=1,
                 amplitude=5,
                 offset=0,
                 phase=0,
                 sample_rate=1000,
                 voltage_range=[-10.0, 10.0],
                 custom_wave=None):

        BaseTask.__init__(self)

        self.voltage_range = voltage_range
        
        buffer_size = sample_rate # if these are not equal, the frequency will be wrong

        #create dev str for various channels
        self.devstr = ""
        if type(channels) is int:
            channels = [channels]
        for channel in channels:
            self.devstr += str(device) + "/ao" + str(channel) + ","
        self.devstr = self.devstr[:-1]        

        # I HAD TO MOVE THIS HERE BECAUSE MKL MESSES WITH WINDOWS API EVENTS
        # LIKE CTRL-C
        import scipy.signal as signal

        ftypes = {
            'sin': amplitude*sin(np.linspace(0,1.0/frequency,
                buffer_size/frequency)*2.0*pi*frequency+phase)+offset,
            'saw': amplitude*signal.sawtooth(np.linspace(0,
                1.0/frequency, buffer_size/frequency)*2.0*pi*frequency+phase)+offset,
            'sqr': amplitude*signal.square(np.linspace(0,1.0/frequency,
                buffer_size/frequency)*2.0*pi*frequency+phase)+offset,
            'custom': custom_wave,
        }

        self.data = ftypes[ftype.lower()]  # create waveform of some sort

        if custom_wave is None:
            for i in range(len(channels)-1):
                self.data = np.column_stack((self.data, self.data))

        self.CreateAOVoltageChan(self.devstr, "", self.voltage_range[0],
                                 self.voltage_range[1], DAQmx_Val_Volts, None)

        self.cfg_sample_clock(rate=sample_rate,
                              edge=DAQmx_Val_Rising,
                              mode='continuous',
                              buffer_size=len(self.data),)

        self.AutoRegisterDoneEvent(0)

        status = self.WriteAnalogF64(len(self.data), 0, -1,
                                     DAQmx_Val_GroupByScanNumber,
                                     self.data, None, None)

    def DoneCallback(self, status):
        """
        Done callback.  Unregistered at this point.  Might just eliminate it.
        """
        #print(status)
        return 0

#---------------------------------------------------------------- Digital Tasks


class DigitalInput(BaseTask):
    '''
    Gets the state of the inputs from the NIDAQ Device/port specified. 

    Parameters
    ----------

    device : 'Dev1'
        String, NIDAQ device id (ex:'Dev1')
    port : 0
        Integer, port number to read data from
    lines : ""
        Line string in nidaq format. Examples: '0', '0:4'
    timeout : 10.0
        Float, seconds to wait for samples

    Returns
    -------

    DigitalInput : Task
        Task object

    Examples
    --------

    >>> task = DigitalInput('Dev1', 0, '0:3') #device 1, port 0, lines 0:3
    >>> task.start()
    >>> data = task.read()
    >>> print(data)
    >>> task.clear()

    '''
    def __init__(self,
                 device='Dev1',
                 port=0,
                 lines="",
                 timeout=10.0):

        BaseTask.__init__(self)

        self.timeout = timeout
        self.port = port
        self.device = device

        if lines is not "":
            if isinstance(lines, int):
                start, end = str(lines), str(lines)
            elif isinstance(lines, str):
                if lines.isdigit():
                    start, end = lines, lines
                else:
                    start, end = lines.split(":")
            self.no_lines = int(end)-int(start)+1
            self.devstr = "{}/port{}/line{}".format(device, port, lines)
        else:
            lines = self.get_input_lines()
            self.no_lines = len(lines)
            self.devstr = str(device) + "/port" + str(port) + "/line0:" + str(self.no_lines-1)
        
        #print self.devstr

        #create channel
        self.CreateDIChan(self.devstr, "", DAQmx_Val_ChanForAllLines)

        self.data = np.zeros(self.no_lines, dtype=np.uint8)

    def get_input_lines(self):
        """
        Returns the number of lines on this port.
        """
        dev = Device(self.device)
        return [l for l in dev.getDILines() if "port{}".format(self.port) in l]

    def read(self):
        """
        Reads the current state of all input lines and returns them as a
            uint8 array.

        Example:
            >>> print di.read()
            [0 1 0 1 0 1 0 1]
        """
        bytesPerSample = c_long()
        samplesPerChannel = c_long()
        self.ReadDigitalLines(1, self.timeout, DAQmx_Val_GroupByChannel,
                              self.data, self.no_lines, samplesPerChannel,
                              bytesPerSample, None)
        return self.data

    def readU32(self):
        """
        Reads the current state of the input lines and returns them as a
            single uint32.
        """
        data = c_ulong()
        self.ReadDigitalScalarU32(self.timeout, data, None)
        return data.value

    def Read(self):
        """
        Deprecated
        """
        return self.read()

    def DoneCallback(self, status):
        """
        Done callback.  Unregistered at this point.  Might just eliminate it.
        """
        print(status)
        return 0 # The function should return an integer


class DigitalInputU32(BaseTask):
    '''
    Like the regular digital input but reads buffers sampled as a specified
        rate.

    Parameters
    ----------

    device : str
        NIDAQ device id (ex:'Dev1')
    lines : int or str
        Lines to reserve and read data from:  32, "0:8"
    timeout : float
        Seconds to wait for samples
    clock_speed : float
        Sample clock speed
    buffer_size : int
        Length of buffer to write to disk
    binary : str
        Binary file to write to

    Returns
    -------

    DigitalInputU32 : Task
        Task object

    Examples
    --------

    >>> task = DigitalInputU32('Dev1', 32) # all 32 lines
    >>> task.start()
    >>> time.sleep(10)  #collect some data
    >>> task.clear()

    '''
    def __init__(self,
                 device='Dev1',
                 lines=32,
                 timeout=10.0,
                 clock_speed=10000.0,
                 buffer_size=1000,
                 binary=None,
                 ):

        BaseTask.__init__(self)

        self.timeout = timeout
        self.lines = lines
        self.device = device
        self.clock_speed = clock_speed
        self.buffer_size = buffer_size
        self.binary = binary

        #set up task properties
        if isinstance(lines, int):
            self.devstr = "%s/line0:%i" % (self.device, lines-1)
        elif isinstance(lines, str):
            self.devstr = "%s/line%s" % (self.device, lines)

        #create channel
        self.CreateDIChan(self.devstr, "", DAQmx_Val_ChanForAllLines)

        #configure sampleclock
        self.cfg_sample_clock(rate=self.clock_speed,
                              edge='rising',
                              mode='continuous',
                              buffer_size=self.buffer_size)

        if self.binary is not None:
            self.outFile = open(self.binary, 'wb')
            self.samples_written = 0
            self.max_samples = 100000*60*60*6

    def cfg_sample_clock(self,
                         rate=10000.0,
                         source="",
                         edge='rising',
                         mode='continuous',
                         buffer_size=1000):
        """
        Custom version of the clock config function.  Needs to re-register
            the NSamples callback.
        """

        # first unregister the old buffer callback if it is registered
        self.unregister_sample_callback()

        # then set up the sample clock
        BaseTask.cfg_sample_clock(self,
                                  rate=rate,
                                  source=source,
                                  edge=edge,
                                  mode=mode,
                                  buffer_size=buffer_size)

        self.data = zeros((buffer_size), dtype=np.uint32)  # data buffer

        # then register the buffer callback
        self.register_sample_callback(buffer_size)


    def EveryNCallback(self):
        """
        Executed every N samples, where N is the buffer_size.  Reads the
            current buffer off of the DAQ.  Writes the samples to disk if
            a binary output file was specified.
        """
        read = int32()
        self.ReadDigitalU32(self.buffer_size, self.timeout, DAQmx_Val_Auto,
                            self.data, self.buffer_size, byref(read), None)
        if self.binary:
            self.outFile.write(self.data.astype(np.uint32).tostring())
            self.samples_written += self.buffer_size
            if self.samples_written > self.max_samples:
                self.stop()
                self.clear()
                raise RuntimeError("Maximum sample count reached.")


    def clear(self):
        """
        Clears the task.  Also flushes and closes the binary file if it
            exists.
        """
        BaseTask.clear(self)
        if self.binary:
            self.outFile.flush()
            self.outFile.close()

    def DoneCallback(self, status):
        """
        Done callback.  Unregistered at this point.  Might just eliminate it.
        """
        #print(status)
        return 0  # The function should return an integer


class DigitalOutput(BaseTask):

    '''
    Sets the current output state of all digital lines.  

    Parameters
    ----------

    device : 'Dev1'
        String, NIDAQ device id (ex:'Dev1')
    port : 0
        Integer, port number to write data to
    lines : "0:3"
        Str, line range in format "start:end" inclusive.  If not provided, all
        lines on the port will be used.
    timeout : 10.0
        Float, seconds for write timeout
    initial_state : 'high'
        String: 'high', 'low' or ndarray for a custom state

    Returns
    -------

    DigitalOutput : Task
        Task object

    Examples
    --------

    >>> task = DigitalOutput('Dev1',1) #device 1, port 1
    >>> lines = len(task.get_output_lines())
    >>> task.start()
    >>> data = np.array([1]*lines,dtype = np.uint8)
    >>> task.write(data)
    >>> task.stop()
    >>> task.clear()

    ##TODO: Can't figure out how to get buffered finite write to work.
        continuous seems ok.

    '''

    def __init__(self, 
                 device='Dev1', 
                 port=0,
                 lines="",
                 timeout=10.0,
                 initial_state=None,
                 ):

        BaseTask.__init__(self)

        self.timeout = timeout
        self.port = port
        self.device = device
        self.initial_state = initial_state

        if lines is not "":
            if isinstance(lines, int):
                start, end = str(lines), str(lines)
            elif isinstance(lines, str):
                if lines.isdigit():
                    start, end = lines, lines
                else:
                    start, end = lines.split(":")
            self.no_lines = int(end)-int(start)+1
            self.devstr = "{}/port{}/line{}".format(device, port, lines)
        else:
            lines = self.get_output_lines()
            self.no_lines = len(lines)
            self.devstr = str(device) + "/port" + str(port) + "/line0:" + str(self.no_lines-1)

        #create IO channel
        self.CreateDOChan(self.devstr, "", DAQmx_Val_ChanForAllLines)

        #create initial state of output lines
        if initial_state is None:
            self.lastOut = self.readLines()
        elif initial_state.lower() == 'low':
            self.lastOut = np.zeros(self.no_lines, dtype=np.uint8)  # keep track of last output #should be gotten from initial state instead
        elif initial_state.lower() == 'high':
            self.lastOut = np.ones(self.no_lines, dtype=np.uint8)
        elif type(initial_state) == np.ndarray:
            self.lastOut = initial_state
        else:
            raise TypeError("Initial state not understood. Try 'high' or 'low'")

    def get_output_lines(self):
        """
        Returns a list of the output lines that are available to this task.
        """
        dev = Device(self.device)
        return [l for l in dev.getDOLines() if "port{}".format(self.port) in l]

    def readLines(self):
        """
        Reads the current state of the output lines and returns them as a
            uint8 array.
        """
        bytes_per_sample = c_long()
        samples_per_channel = c_long()
        no_lines = self.no_lines
        buf = np.zeros(no_lines, dtype=np.uint8)
        self.ReadDigitalLines(1, self.timeout,
                              DAQmx_Val_GroupByChannel, buf, no_lines,
                              samples_per_channel, bytes_per_sample,
                              None)
        return buf

    def readU32(self):
        """
        Reads the current state of the output lines and returns them as a
            single uint32.
        """
        data = c_ulong()
        self.ReadDigitalScalarU32(self.timeout, data, None)
        return data.value

    def write(self, data, samples_per_line=1, autostart=1):
        '''Writes a numpy array of data to set the current output state

        Parameters
        ----------

        data : required
            ndarray, uint8, cols should be the number of lines. rows should be
            number of samples per channel
        samples_per_line: 1
            int, number of samples to write per line

        Returns
        -------
        None

        Examples
        --------

        >>> data = np.array([1,0,1,0],dtype=np.uint8)
        >>> task.Write(data, 1)

        ##TODO: Automatically calculate samples per line based on array shape.

        '''

        status = self.WriteDigitalLines(samples_per_line,
                                        autostart,
                                        self.timeout,
                                        DAQmx_Val_GroupByChannel,
                                        data,
                                        None,
                                        None)
        if samples_per_line == 1:
            self.lastOut = data
        else:
            self.lastOut = data[-1]
        return status

    def writeU8(self, data, groupby='c', autostart=0):
        """
        Writes a U8 ndarray to the digital channel.

        Args:
            data (numpy.ndarray): the data to write
            groupby (int or str): grouping type.
            autostart (int): whether to attempt to autostart.

        """
        status = self.WriteDigitalU8(len(data),
                                     autostart,
                                     self.timeout,
                                     get_group_val(groupby),
                                     data,
                                     None,
                                     None)
        return status

    def writeU16(self, data, groupby='c', autostart=0):
        """
        Writes a U16 ndarray to the digital channel.

        Args:
            data (numpy.ndarray): the data to write
            groupby (int or str): grouping type.
            autostart (int): whether to attempt to autostart.

        """
        status = self.WriteDigitalU16(len(data),
                                      autostart,
                                      self.timeout,
                                      get_group_val(groupby),
                                      data,
                                      None,
                                      None)
        return status

    def writeU32(self, data, groupby='c', autostart=0):
        """
        Writes a U32 ndarray to the digital channel.

        Args:
            data (numpy.ndarray): the data to write
            groupby (int or str): grouping type.
            autostart (int): whether to attempt to autostart.

        """
        status = self.WriteDigitalU32(len(data),
                                      autostart,
                                      self.timeout,
                                      get_group_val(groupby),
                                      data,
                                      None,
                                      None)
        return status

    def Write(self, data, samples_per_line=1):
        """
        Deprecated
        """
        return self.write(data, samples_per_line)

    def writeBit(self, index, value):
        '''
        Writes a single bit to the given line index.

        Parameters
        ----------
        index : int
            Line (bit) to write to
        value : int
            Value to write (1 or 0)

        Returns
        -------
        None

        Examples
        --------

        >>> task.writeBit(0,0)  # sets line 0 to 1

        '''
        self.lastOut[index] = value
        status = self.WriteDigitalLines(1, 1, self.timeout,
                                        DAQmx_Val_GroupByChannel,
                                        self.lastOut, None, None)
        return status

    def WriteBit(self, index, value):
        """
        Deprecated
        """
        return self.writeBit(index, value)

    def cfg_sample_clock(self,
                         freq,
                         source="",
                         edge='rising',
                         mode='continuous',
                         buffer_size=10000,
                         ):
        """
        For digital output tasks, we cannot imply a source, so I provide a
            default option that usually works if the user doesn't supply one.
        """
        if not source:
            source = "/{}/do/SampleClockTimebase".format(self.device)
        status = BaseTask.cfg_sample_clock(self,
                                           freq,
                                           source,
                                           edge,
                                           mode,
                                           buffer_size,)
        return status


class EventInput(BaseTask):
    """
    Takes a list of digital lines and registers a NI task with an asynchronous
        event callback for change events on those lines.

    Parameters
    ----------
    device : str
        Device ID for NIDAQ board.  Default "Dev1"
    bits : int 
        How many lines to read. Default 32
    event_lines : list of ints
        Which lines should trigger a change event.  Default [0]
    timeout : float
        Timeout for a single read in seconds
    buffer_size : int
        Size of buffer to before triggering callback (doesn't do anything)
    buffer_callback : callable
        Custom callback function for a full buffer.
    force_synchronous_callback : bool
        Force callback in aquisition thread to ensure accuracy.

    """
    def __init__(self,
                 device='Dev1',
                 bits=32,
                 event_lines=[0],
                 timeout=0.1,
                 buffer_size=200,
                 buffer_callback=None,
                 force_synchronous_callback=False,
                 ):
        BaseTask.__init__(self)
        self.timeout = timeout
        self.device = device
        self.event_lines = event_lines  #
        self.timeout = timeout  #should this even have a timeout?
        
        self.bits = bits
        self.data = c_ulong()

        self.devstr = "%s/line0:%i" % (self.device, bits-1)

        self.CreateDIChan(self.devstr, "", DAQmx_Val_ChanForAllLines)

        self.CfgChangeDetectionTiming(self.devstr, self.devstr,
                                      DAQmx_Val_HWTimedSinglePoint, 1)
        if force_synchronous_callback:
            # works but allegedly hurts performance of the thread that calls it
            # probably necessary for precision though? Talking to NI about it
            # honestly can't find a performance difference as long as I thread
            # when you use synchronous callbacks
            options = DAQmx_Val_SynchronousEventCallbacks
        else:
            options = 0

        self.AutoRegisterSignalEvent(DAQmx_Val_ChangeDetectionEvent, options)

        #Allow for custom buffer callbacks
        if not buffer_callback:
            self.buffer_callback = self.print_data
        else:
            self.buffer_callback = buffer_callback

        #self.sample_count = 0
        self.events = 0

        self.timeouts = []

    #@timeit
    def SignalCallback(self):
        """
        Event callback for rising/falling edges.
        """
        try:
            data = self.read()
            self.buffer_callback(data)
        except Exception as e:
            #hrrmm what is this about
            print(e)
            self.timeouts.append([self.events, self.data])
        self.events += 1

    def read(self):
        self.ReadDigitalScalarU32(self.timeout, self.data, None)
        return self.data

    def print_data(self, data):
        print(data)


class CounterInputU32(BaseTask):
    """
    Generic edge counter for single U32 counter.

    Parameters
    ----------
    device : str
        NI DAQ ID.  Ex: "Dev1"
    counter : str
        Counter terminal.  Ex: 'ctr0'
    edge : str
        Edge to count.  Either "rising" or "falling"
    direction : str
        Counter direction.  'up' or 'down'
    initial_count : int
        Initial counter value.
    timeout: float
        Read timeout.

    """
    def __init__(self,
                 device='Dev1',
                 counter='ctr0',
                 edge='rising',
                 direction='up',
                 initial_count=0,
                 timeout=10.0,
                 ):

        BaseTask.__init__(self)
        self.device = device
        self.counter = counter
        self.edge = edge
        self.direction = direction
        self.initial_count = initial_count
        self.buffer_size = None
        self.timeout = timeout

        self.devstr = "%s/%s" % (device, counter)

        if direction.lower() == 'up':
            dir_val = DAQmx_Val_CountUp
        elif direction.lower() == 'down':
            dir_val = DAQmx_Val_CountDown
        else:
            raise KeyError("Invalid direction.  Try 'up' or 'down'.")

        self.CreateCICountEdgesChan(self.devstr, "", get_edge_val(self.edge),
                                    initial_count, dir_val)

        self.data = c_ulong()

    def read(self):
        """
        A simple scalar read of the current counter value.
        """
        self.ReadCounterScalarU32(self.timeout, self.data, None)
        return self.data

    def setup_file_output(self,
                          path=None,
                          file_type="bin",
                          buffer_size=1000,
                          ):
        """
        Sets up data output writing.  This alone is insufficient.  You must Also
            configure the sample clock.
        """
        if not path:
            self.buffer_count = None
            self.unregister_sample_callback()
            return True
        else:
            self.buffer_size = buffer_size
            self.buffer_count = 0
            self.register_sample_callback(self.buffer_size)

        self.data = np.zeros(self.buffer_size, dtype=np.uint32)
        
        if file_type == 'bin':
            self.output_file = open(path, 'wb')
        else:
            raise NotImplementedError("file types other than binary are unimplemented.")

    def EveryNCallback(self):
        """
        Callback for buffer read.  Occurs automatically when `self.buffer_size`
            samples are read if buffered reading is enabled.
        """
        try:
            read = int32()

            # read into the data buffer
            self.ReadCounterU32(self.buffer_size, self.timeout, self.data,
                                self.buffer_size, byref(read), None)

            #self.ReadDigitalU32(self.buffer_size, self.timeout, DAQmx_Val_Auto,
            #        self.data, self.buffer_size, byref(read), None)

            self.output_file.write(self.data.tostring())
            self.buffer_count += 1
        except Exception as e:
            print("Failed to read buffer #%i -> %s" % (self.buffer_count, e))

    def getCountEdgesTerminal(self):
        """
        Returns the terminal for edge counting input (str)

        Example output: "/Dev1/PFI8"
        """
        buffer_size = 1024
        lines = " "*buffer_size
        self.GetCICountEdgesTerm(self.devstr, lines, buffer_size)
        return lines.strip().strip('\x00').split(', ')

    def setCountEdgesTerminal(self, terminal):
        """
        Sets the edge counting input terminal.

        Example input: "Ctr0InternalOutput"
        """
        self.SetCICountEdgesTerm(self.devstr, terminal)

    # def clear(self):
    #     super(CounterInputU32, self).clear()
    #     if self.output_file:
    #         self.output_file.close()

    # def stop(self):
    #     super(CounterInputU32, self).stop()
    #     if self.output_file:
    #         self.output_file.close()


class CounterInputU64(object):
    """
    Counter32 pair cascaded into a 64-bit counter.

    The LSB counter counts each edge.  The MSB counter counts the LSB counter's
        rollover.

    Parameters
    ----------
    device : str
        NI DAQ ID.  Ex: "Dev1"
    lsb_counter : str
        Counter terminal for least significant bits.  Ex: 'ctr0'
    msb_counter : str
        Counter terminal for most significant bits. Ex: 'ctr1'
    edge : str
        Edge to count.  Either "rising" or "falling"
    direction : str
        Counter direction.  'up' or 'down'
    initial_count : int
        Initial counter value.

    """
    def __init__(self,
                 device='Dev1',
                 lsb_counter='ctr0',
                 msb_counter='ctr1',
                 edge='rising',
                 direction='up',
                 initial_count=0,
                 timeout=10.0,
                 ):

        self.device = device
        self.lsb_counter = lsb_counter
        self.msb_counter = msb_counter
        self.edge = edge
        self.direction = direction
        self.initial_count = initial_count
        self.timeout = timeout

        #configure least significant bit counter
        self.lsb = CounterInputU32(device=self.device,
                                   counter=self.lsb_counter,
                                   edge=self.edge,
                                   direction=self.direction,
                                   timeout=self.timeout,)

        #configure most significant bit counter
        #counts LSB counter rollover
        msb_term = "Ctr%sInternalOutput" % self.lsb_counter[-1]
        self.msb = CounterInputU32(device=self.device,
                                   counter=self.msb_counter,
                                   edge=self.edge,
                                   direction=self.direction,
                                   timeout=self.timeout,)
        self.msb.setCountEdgesTerminal(msb_term)

        #keep track of previous lsb value
        self.lsb_data = c_ulong(0)

    def read(self):
        """
        Reads both counters and returns the values as unsigned 32-bit integers.
            We have to read MSB twice if the new LSB data is smaller than the
            previous read (which occurs when the LSB rolls over), in case the
            MSB changed between the reads.
        """
        ##TODO: IS THIS THE FASTEST WAY TO DO THIS?
        self.msb_data = self.msb.read()
        lsb_data = self.lsb.read()
        if lsb_data < self.lsb_data:
            self.msb_data = self.msb.read()
        self.lsb_data = lsb_data
        return (self.lsb_data, self.msb_data)

    def getCountEdgesTerminal(self):
        """
        Returns the terminals for both edge counters.
        """
        return [self.lsb.getCountEdgesTerminal(),
                self.msb.getCountEdgesTerminal()]

    def setCountEdgesTerminal(self, terminal):
        """
        Sets the terminals for the LSB counter.  The MSB counter terminal has
            to be the output of the LSB counter so we don't change it.
        """
        self.lsb.setCountEdgesTerminal(terminal)

    def start(self):
        """
        Start counter task.
        """
        self.msb.start()
        self.lsb.start()

    def stop(self):
        """
        Stop counter task.
        """
        self.lsb.stop()
        self.msb.stop()

    def clear(self):
        """
        Clear counter task.  It cannot be restarted.
        """
        try:
            self.stop()
        except:
            ##TODO: type this exception
            pass
        self.lsb.clear()
        self.msb.clear()


class CounterOutputFreq(BaseTask):
    """
    Generic counter pulse output.  Specified in terms of frequency.
    See: http://zone.ni.com/reference/en-XX/help/370471Y-01/daqmxcfunc/daqmxcreatecopulsechanfreq/

    Parameters
    ----------
    device : str
        NI DAQ ID.  Ex: "Dev1"
    counter : str
        Counter terminal.  Ex: 'ctr0'
    init_delay : float
        Initial delay in seconds after task start.
    freq : float
        Pulse frequency in Hz
    duty_cycle : float
        Duty cycle of pulse (Between 0.0 and 1.0)
    idle_state : str
        Idle state of pulse.  'high' or 'low'
    timing : str
        Timing type. 'hw' or 'sw'
    buffer_size : int
        Buffer size (Only applies to software timing)

    """
    def __init__(self,
                 device="Dev1",
                 counter='ctr0',
                 init_delay=0.0,
                 freq=500000.0,
                 duty_cycle=0.50,
                 idle_state='low',
                 timing='hw',
                 buffer_size=1000000,
                 ):

        BaseTask.__init__(self)
        self.device = device
        self.counter = counter
        self.init_delay = init_delay
        self.freq = freq
        self.duty_cycle = duty_cycle
        self.buffer_size = buffer_size
        self.timing = timing
        self.idle_state = idle_state

        self.devstr = "%s/%s" % (device, counter)

        self.idle_state = get_elevation_val(self.idle_state)

        self.CreateCOPulseChanFreq(self.devstr,
                                   "",
                                   DAQmx_Val_Hz,
                                   self.idle_state,
                                   init_delay,
                                   freq,
                                   duty_cycle)

        if timing.lower() == "hw":
            #assuming 20MHzTimebase is ok for now
            # see http://zone.ni.com/reference/en-XX/help/370466W-01/mxcncpts/termnames/
            self.CfgSampClkTiming("20MHzTimebase",
                                  20000000.0,
                                  DAQmx_Val_Rising,
                                  DAQmx_Val_HWTimedSinglePoint,
                                  0)
        elif timing.lower() == 'sw':
            #higher buffers size for higher freq output
            #see http://zone.ni.com/reference/en-XX/help/370466V-01/mxcncpts/buffersize/
            self.CfgImplicitTiming(DAQmx_Val_ContSamps, buffer_size)
        else:
            raise KeyError("Timing type is not supported.  Try 'hw' or 'sw'")

    def getPulseTerminal(self):
        """
        Returns the terminal for Pulse output (str)

        Example output: "/Dev1/PFI13"
        """
        buffer_size = 1024
        lines = " "*buffer_size
        self.GetCOPulseTerm(self.devstr,
                            lines,
                            buffer_size)
        return lines.strip().strip('\x00').split(', ')


class CounterInputPWM(BaseTask):
    """
    Pulse Width Counter.
    """
    def __init__(self,
                 device='Dev1',
                 counter='ctr0',
                 starting_edge='rising',
                 min_val=0.0,
                 max_val=1.0,
                 units='seconds',
                 timeout=10.0,
                 ):
        BaseTask.__init__(self)
        self.device = device
        self.counter = counter
        self.devstr = "%s/%s" % (device, counter)
        self.starting_edge = get_edge_val(starting_edge)
        self.min_val = min_val
        self.max_val = max_val
        self.units = units.lower()
        self.timeout = timeout

        if self.units == 'seconds':
            units_val = DAQmx_Val_Seconds
        elif self.units == 'timebase':
            units_val = DAQmx_Val_Ticks
        else:
            raise KeyError("Invalid edge type.  Try 'seconds' or 'timebase'")

        self.CreateCIPulseWidthChan(self.devstr, "", self.min_val, self.max_val,
                                    units_val, self.starting_edge, None)

        self.data = c_double()

    def read(self):
        """
        Returns the current high width.
        """
        self.ReadCounterScalarF64(self.timeout, self.data, None)
        return self.data

    def setPulseWidthTerminal(self, terminal):
        """
        Sets to input terminal to measure pulse width.
        """
        self.SetCIPulseWidthTerm(self.devstr, terminal)


def syncit(digital_out_task, line, invert=False):
    """
    Function or method decorator that can be used for synchronization.  It sets
        the digital line high or low during the decorated function call.

    Parameters
    ----------
    digital_out_task : DigitalOutput object
        DigitalOutput object that you'd like to use to send the signal.
    line : int
        Which line to send the signal on.
    invert : bool (False)
        Whether to invert the signal.

    Returns
    -------
    None (This is a decorator)

    Examples
    --------
    >>> import time
    >>> do = DigitalOutput("Dev1", 0)
    >>> do.StartTask()

    >>> @syncit(do, 0)
    >>> def do_stuff():
    ...     time.sleep(1)

    >>> do_stuff()
    >>> do.ClearTask()

    """

    do = digital_out_task
    if invert:
        high, low = 0, 1
    else:
        high, low = 1, 0

    def wrap(f):
        def synced(*args, **kwargs):
            do.WriteBit(line, high)
            result = f(*args, **kwargs)
            do.WriteBit(line, low)
            return result
        return synced
    return wrap

def get_edge_val(edge):
    """
    Gets the correct edge constant for a given input.
    """
    if edge in [DAQmx_Val_Rising, DAQmx_Val_Falling]:
        pass
    elif isinstance(edge, str):
        if edge.lower() in ["falling", 'f']:
            edge = DAQmx_Val_Falling
        elif edge.lower() in ["rising", 'r']:
            edge = DAQmx_Val_Rising
        else:
            raise ValueError("Only 'rising'('r') or 'falling'('f') is accepted.")
    else:
        raise ValueError("Edge must be str ('falling') or int (DAQmx_Val_Falling)")
    return edge

def get_mode_val(mode):
    """
    Gets the correct mode constant for a given input.
    """
    if mode in [DAQmx_Val_FiniteSamps,
                DAQmx_Val_ContSamps,
                DAQmx_Val_HWTimedSinglePoint]:
        pass
    elif isinstance(mode, str):
        if mode.lower() in ["finite", 'f']:
            mode = DAQmx_Val_FiniteSamps
        elif mode.lower() in ["continuous", 'c']:
            mode = DAQmx_Val_ContSamps
        elif mode.lower() in ['hwtsp', 'h']:
            mode = DAQmx_Val_HWTimedSinglePoint
        else:
            raise ValueError("Only 'finite'('f'), 'continuous'('c'), or 'hwtsp'('h') is accepted.")
    else:
        raise ValueError("Mode must be str ('finite') or int (DAQmx_Val_FiniteSamps)")
    return mode

def get_group_val(group):
    """
    Gets the correct grouping type for a given input.
    """
    if group in [DAQmx_Val_GroupByChannel,
                DAQmx_Val_GroupByScanNumber]:
        pass
    elif isinstance(group, str):
        if group.lower() in ['c', 'channel', 'chan']:
            group = DAQmx_Val_GroupByChannel
        elif group.lower() in ['s', 'scan', 'scannumber']:
            group = DAQmx_Val_GroupByScanNumber
        else:
            raise ValueError("Only 'channel'('c') or 'scan'('s') is accepted.")
    else:
        raise ValueError("Mode must be str ('channel') or int (DAQmx_Val_GroupByChannel)")
    return group

def get_direction_val(direction):
    """
    Gets the correct direction type for a given input.
    """
    if direction in [DAQmx_Val_Acquired_Into_Buffer,
                     DAQmx_Val_Transferred_From_Buffer,]:
        pass
    elif isinstance(direction, str):
        if direction.lower() in ['in', 'input', 'acquired', 'acq', 'i']:
            direction = DAQmx_Val_Acquired_Into_Buffer
        elif direction.lower() in ['out', 'output', 'written', 'o']:
            direction = DAQmx_Val_Transferred_From_Buffer
        else:
            raise ValueError("Only 'input'('i') or 'output'('o') is accepted.")
    else:
        raise ValueError("Direction must be str ('input') or int (DAQmx_Val_Transferred_From_Buffer).")
    return direction

def get_synchronous_val(synchronous):
    """
    Gets the correct synchronous type for a given input.
    """
    if synchronous in [0, DAQmx_Val_SynchronousEventCallbacks]:
        pass
    elif synchronous is True:
        synchronous = DAQmx_Val_SynchronousEventCallbacks
    elif synchronous is False:
        synchronous = 0
    else:
        raise ValueError("Synchronous must be bool or int (DAQmx_Val_SynchronousEventCallbacks)")
    return synchronous

def get_elevation_val(high_or_low):
    """
    Gets the correct elevation value for a given input.
    """
    if high_or_low in [DAQmx_Val_Low,
                       DAQmx_Val_High,]:
        pass
    elif isinstance(high_or_low, str):
        if high_or_low.lower() in ['high', 'h']:
            high_or_low = DAQmx_Val_High
        elif high_or_low.lower() in ['low', 'l']:
            high_or_low = DAQmx_Val_Low
        else:
            raise ValueError("Only 'high' 'h' or 'low' 'l' is accepted.")
    else:
        raise ValueError("Elevation must be str ('high') or int (DAQmx_Val_High)")
    return high_or_low

def get_input_terminal_config(config):
    """
    Gets the correct config value for a given input.
    """
    if config in [DAQmx_Val_Cfg_Default,
                  DAQmx_Val_RSE,
                  DAQmx_Val_NRSE,
                  DAQmx_Val_Diff,
                  DAQmx_Val_PseudoDiff,]:
        pass
    elif isinstance(config, str):
        config = config.lower()
        if config in ['default',]:
            config = DAQmx_Val_Cfg_Default
        elif config in ['rse', 'r']:
            config = DAQmx_Val_RSE
        elif config in ['nrse', 'n']:
            config = DAQmx_Val_NRSE
        elif config in ['diff', 'd']:
            config = DAQmx_Val_Diff
        elif config in ['pseudodiff', 'pseudo', 'p']:
            config = DAQmx_Val_PseudoDiff
        else:
            raise ValueError("Invalid terminal config type. Try 'rse' or 'diff'.")
    else:
        raise ValueError("Terminal config type must be str ('rse') or int (DAQmx_Val_Diff).")
    return config

#---------------------------------------------------------------------- if main
if __name__ == '__main__':
    import time

    freq = 100000.0
    file_name = "C:/test_counter.bin"

    ci = CounterInputU32("Dev2",
                         "ctr0")
    co = CounterOutputFreq("Dev2",
                           'ctr2',
                           freq=freq)

    ci.setup_file_output(file_name)
    ci.cfg_sample_clock(rate=freq,
                        #source="Ctr2InternalOutput",
                        source="di/SampleClock",
                        )
    ci.setCountEdgesTerminal("Ctr2InternalOutput")

    co.start()
    ci.start()

    time.sleep(2)

    ci.stop()
    co.stop()

    ci.clear()
    co.clear()

    time.sleep(2)

    import numpy as np
    data = np.fromfile(file_name, dtype=np.uint32)
    print data