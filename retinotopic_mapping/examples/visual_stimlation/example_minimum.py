# -*- coding: utf-8 -*-
"""
the minimum script to run 10 seconds of black screen
"""

import matplotlib.pyplot as plt
import retinotopic_mapping.StimulusRoutines as stim
from retinotopic_mapping.MonitorSetup import Monitor, Indicator
from retinotopic_mapping.DisplayStimulus import DisplaySequence

# Initialize Monitor object
mon = Monitor(resolution=(1200, 1920), dis=15., mon_width_cm=52., mon_height_cm=32.)

# Initialize Inicator object
ind = Indicator(mon)

# Initialize DisplaySequence object
ds = DisplaySequence(log_dir='C:/data')

# Initialize UniformContrast object
uc = stim.UniformContrast(monitor=mon, indicator=ind, duration=10., color=-1.)

# set uniform contrast stimulus into the DisplaySequence object
ds.set_stim(uc)

# start display
ds.trigger_display()

# plot distribution of frame duration
plt.show()