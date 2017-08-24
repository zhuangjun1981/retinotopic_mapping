import numpy as np
import time
from toolbox.IO.nidaq import DigitalOutput, CounterOutputFreq

# Digital Output requires using a counter or analog input/output clock as the clock source
# Toggle D0-7 once per second

if __name__ == '__main__':
    clockFreq = 10

    co = CounterOutputFreq('Dev1', counter='ctr0', freq=clockFreq)
    co.start()

    do = DigitalOutput('Dev1', port=0)
    do.cfg_sample_clock(clockFreq, mode='f', buffer_size=clockFreq,
                        source='/Dev1/ctr0InternalOutput')

    buf = np.array([0]*clockFreq, dtype=np.uint32)
    buf[:clockFreq/2] = 255

    while True:
        do.writeU32(buf, autostart=0)
        do.start()
        do.WaitUntilTaskDone(10)
        do.stop()
        print time.clock()


