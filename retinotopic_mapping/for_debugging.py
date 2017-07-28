# -*- coding: utf-8 -*-
"""
Example script to test that everything is working. Running this script is a
good first step for trying to debug your experimental setup and is also a
great tool to familiarize yourself with the parameters that are used to
generate each specific stimulus.

!!!IMPORTANT!!!
Note that once you are displaying stimulus, if you want to stop the code from
running all you need to do is press either one of the 'Esc' or 'q' buttons.
"""

import numpy as np
import StimulusRoutines as stim
from MonitorSetup import Monitor, Indicator
from DisplayStimulus import DisplaySequence

"""
To get up and running quickly before performing any experiments it is 
sufficient to setup two monitors -- one for display and one for your python 
environment. If you don't have two monitors at the moment it is doable with
only one. 

Edit the following block of code with your own monitors respective parameters.
Since this script is for general debugging and playing around with the code, 
we will arbitrarily populate variables that describe the geometry of where 
the mouse will be located during an experiment. All we are interested in 
here is just making sure that we can display stimulus on a monitor and learning
how to work with the different stimulus routines.
"""
#==============================================================================
resolution = (1280,1024) #enter your monitors resolution
mon_width_cm = 38 #enter your monitors width in cm
mon_height_cm = 40 #enter your monitors height in cm
refresh_rate = 60  #enter your monitors height in Hz
#==============================================================================
# The following variables correspond to the geometry of the mouse with
# respect to the monitor, don't worry about them for now we just need them
# for all of the functions to work

C2T_cm = mon_height_cm / 2.
C2A_cm = mon_width_cm / 2.
mon_tilt = 26.56
dis = 15.

# Set the downsample rate; needs to be an integer `n` such that each resolution
# number is divisble by `n`,
downsample_rate = 4

# Initialize the monitor and ind objects
mon = Monitor(resolution=resolution,
            dis=dis,
            mon_width_cm=mon_width_cm,
            mon_height_cm=mon_height_cm,
            C2T_cm=C2T_cm,
            C2A_cm=C2A_cm,
            mon_tilt=mon_tilt,
            downsample_rate=downsample_rate)
ind = Indicator(mon)

dg = stim.DriftingGratingCircle(mon,ind,tf_list=(4.,8.),iteration=2)
a = dg.generate_frames_by_index()
b = dg.generate_frames()
""" Now for the fun stuff! Each block of code below shows an example of
the stimulus routines that are currently implemented in the codebase. Uncomment
each block and run the script to view the stimulus presentations. This is where
you might need to start debugging!
"""
#========================== Uniform Contrast Stimulus =========================
#uniform_contrast = stim.UniformContrast(mon,
#                                        ind,
#                                        duration=10.,
#                                        color=0.)
#ds = DisplaySequence(log_dir=r'C:\data',
#                     backupdir=None,
#                     display_iter=2,
#                     is_triggered=False,
#                     is_sync_pulse=False,
#                     display_screen=1)
#
#ds.set_stim(uniform_contrast)
#ds.trigger_display()
#==============================================================================


#======================= Flashing Circle Stimulus =============================
#flashing_circle = stim.FlashingCircle(mon,
#                                      ind,
#                                      radius=20.,
#                                      flash_frame=10)
#ds = DisplaySequence(log_dir=r'C:\data',
#                     backupdir=None,
#                     is_triggered=False,
#                     is_sync_pulse=False,
#                     by_index=False,
#                     display_iter=2,
#                     display_screen=1)
#ds.set_stim(flashing_circle)
#ds.trigger_display()
#==============================================================================


#======================== Sparse Noise Stimulus ===============================
#sparse_noise = stim.SparseNoise(mon,
#                                ind,
#                                subregion=(-20.,20.,40.,60.),
#                                grid_space=(10, 10),
#                                background=0.,
#                                sign='ON')
#ds = DisplaySequence(log_dir=r'C:\data',
#                     backupdir=r'C:\data',
#                     is_triggered=False,
#                     display_iter=2,
#                     display_screen=1)
#ds.set_stim(sparse_noise)
#ds.trigger_display()
#==============================================================================

#======================= Sparse Noise pt 2 ====================================
#sparse_noise = stim.SparseNoise(mon,ind)
#ds = DisplaySequence(log_dir=r'C:\data',
#                     backupdir=None,
#                     is_triggered=False,
#                     is_sync_pulse=False,
#                     display_screen=1)
#ds.set_stim(sparse_noise)
#ds.trigger_display()
#==============================================================================


#======================= Drifting Grating Circle Stimulus =====================
dg = stim.DriftingGratingCircle(mon,
                                              ind,
                                              sf_list=(0.08,0.16),
                                              tf_list=(4.0,8.0),
                                              dire_list=(0.,),
                                              con_list=(0.5,1.),
                                              size_list=(5.,10.))
#ds = DisplaySequence(log_dir=r'C:\data',
#                     backupdir=None,
#                     display_iter = 2,
#                     is_triggered=False,
#                     is_sync_pulse=False,
#                     is_interpolate=False,
#                     display_screen=1)
#ds.set_stim(drifting_grating)
#ds.trigger_display()

#======================== Drifting Grating pt 2 ===============================
#drifting_grating2 = stim.DriftingGratingCircle(mon,
#                                               ind,
#                                               center=(60.,0.),
#                                               sf_list=[0.08, 0.16],
#                                               tf_list=[4.,2.],
#                                               dire_list=[np.pi/6],
#                                               con_list=[1.,0.5],
#                                               size_list=[40.],
#                                               block_dur=2.,
#                                               pregap_dur=2.,
#                                               postgap_dur=3.,
#                                               midgap_dur=1.)
#
#ds=DisplaySequence(log_dir=r'C:\data',
#                   backupdir=None,
#                   display_iter = 2,
#                   is_triggered=False,
#                   is_sync_pulse=False,
#                   is_interpolate=False,
#                   display_screen=1)
#ds.set_stim(drifting_grating2)
#ds.trigger_display()
#==============================================================================


#===================== Kalatsky&Stryker Stimulus ==============================
#KS_stim = stim.KSstim(mon,
#                    ind,
#                    coordinate='degree',
#                    sweep_frame=1,
#                    flicker_frame=100)
#
#ds = DisplaySequence(log_dir=r'C:\data',
#                     backupdir=None,
#                     is_triggered=False,
#                     display_iter=2,
#                     display_screen=1)
#ds.set_stim(KS_stim)
#ds.trigger_display()
#==============================================================================

#======================= Kalatsky&Stryker pt 2 ================================
#KS_stim_all_dir = stim.KSstimAllDir(mon,ind,step_width=0.3)
#ds = DisplaySequence(log_dir=r'C:\data',
#                     backupdir=None,
#                     display_iter = 2,
#                     is_triggered=False,
#                     is_sync_pulse=False,
#                     display_screen=1)
#ds.set_stim(KS_stim_all_dir)
#ds.trigger_display()
#==============================================================================