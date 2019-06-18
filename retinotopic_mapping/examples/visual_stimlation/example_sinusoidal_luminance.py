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
sl = stim.SinusoidalLuminance(monitor=mon, indicator=ind,
                              pregap_dur=1.,
                              midgap_dur=0.,
                              postgap_dur=3.,
                              max_level=0.,
                              min_level=-0.8,
                              frequency=0.5, cycle_num=3,
                              start_phase=0.)

# set uniform contrast stimulus into the DisplaySequence object
ds.set_stim(sl)

# start display
ds.trigger_display()

# plot distribution of frame duration
plt.show()