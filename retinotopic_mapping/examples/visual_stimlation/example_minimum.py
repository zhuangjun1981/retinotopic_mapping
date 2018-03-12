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
ind = Indicator(mon)
ds = DisplaySequence(log_dir='C:/data')
uc = stim.UniformContrast(monitor=mon, indicator=ind, duration=10., color=-1.)
ds.set_stim(uc)
ds.trigger_display()
plt.show()