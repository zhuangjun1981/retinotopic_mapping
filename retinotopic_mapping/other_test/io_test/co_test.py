# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 16:40:57 2015

@author: derricw
"""
import time
import logging
from toolbox.IO.nidaq import *


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    device = "Dev2"

    counters = GetCOChannels(device)
    print("Available Counters:")
    print(counters)

    if counters:

        co = CounterOutputFreq(device,
                               "ctr0",
                               init_delay=0.0,
                               freq=1000000.0,
                               duty_cycle=0.25,
                               idle_state='low',
                               timing="hw",
                               buffer_size=1000000)

        ci = CounterInputPWM(device,
                             "ctr2",
                             starting_edge='falling',
                             min_val=20.0e-9,
                             max_val=20.0e-3,
                             units='seconds',
                             timeout=10.0,)

        ci.SetCIPulseWidthTerm("Dev2/ctr2", "Ctr0InternalOutput")

        ci.start()

        co.start()

        time.sleep(1.0)

        for i in range(100):
            print ci.read().value

        time.sleep(1.0)

        co.stop()

        ci.stop()