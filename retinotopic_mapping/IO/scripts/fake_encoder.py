"""
Creates a fake analog encoder signal using an analog output channel.

Examples:

    $> python fake_encoder.py -o 0 -d Dev1

Where -o is the output channel and -d is the NI device.

"""

from toolbox.IO.nidaq import AnalogOutput
import time
import signal
import sys
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-o",
                        "--output_channel",
                        type=int,
                        help="Analog output channel for 5V power.",
                        default=0)
    parser.add_argument("-d", "--device", type=str, default="Dev1",
                        help="NI DAQ Device ID")

    args = parser.parse_args()

    device = args.device
    output_channel = args.output_channel

    ao = AnalogOutput(device, channels=[output_channel])

    def exit_handler(*args):
        print("Closing...")
        ao.clear()
        sys.exit(0)

    signal.signal(signal.SIGINT, exit_handler)

    ao.start()

    print("\nStarted fake encoder output on {}, output channel {}".format(
        device, output_channel))
    print("#"*30)

    while True:
        for i in range(1000):
            data = np.array([5.0*i/1000], dtype=np.float64)
            ao.write(data)
            sys.stdout.write("\r  {:<10} Volts".format(float(data)))
            sys.stdout.flush()
            time.sleep(0.01)
