"""
Simple test of the base task.

1. Creates a BaseTask
2. Configures it for digital output.
3. Starts the task.
4. Creates an output buffer.
5. Writes the output buffer to the digital lines.
6. Stops the task.
7. Clears the task.

"""



from toolbox.IO.nidaq import BaseTask
from PyDAQmx.DAQmxConstants import *
import numpy as np

if __name__ == '__main__':
    b = BaseTask()

    b.CreateDOChan('Dev1/port0/line0:4',
                   '',
                   DAQmx_Val_ChanForAllLines,)

    b.start()
    buf = np.array([0,1,0,1], dtype=np.uint8)
    b.WriteDigitalLines(1, 0, 10.0, DAQmx_Val_GroupByChannel, buf,
                         None, None)
    b.stop()
    b.clear()