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
import matplotlib.pyplot as plt
import retinotopic_mapping.StimulusRoutines as stim
from retinotopic_mapping.MonitorSetup import Monitor, Indicator
from retinotopic_mapping.DisplayStimulus import DisplaySequence

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
resolution = (1200,1920) #enter your monitors resolution
mon_width_cm = 52 #enter your monitors width in cm
mon_height_cm = 32 #enter your monitors height in cm
refresh_rate = 60  #enter your monitors height in Hz
#==============================================================================
# The following variables correspond to the geometry of the mouse with
# respect to the monitor, don't worry about them for now we just need them
# for all of the functions to work

C2T_cm = mon_height_cm / 2.
C2A_cm = mon_width_cm / 2.
mon_tilt = 30.
dis = 15.

# Set the downsample rate; needs to be an integer `n` such that each resolution
# number is divisble by `n`,
downsample_rate = 5

# Initialize the monitor and ind objects
mon = Monitor(resolution=resolution, dis=dis, mon_width_cm=mon_width_cm, mon_height_cm=mon_height_cm,
              C2T_cm=C2T_cm, C2A_cm=C2A_cm, mon_tilt=mon_tilt, downsample_rate=downsample_rate)
# mon.plot_map()
# plt.show()
ind = Indicator(mon, width_cm = 3., height_cm = 3., position = 'northeast', is_sync = True, freq = 2.)

""" Now for the fun stuff! Each block of code below shows an example of
the stimulus routines that are currently implemented in the codebase. Uncomment
each block and run the script to view the stimulus presentations. This is where
you might need to start debugging!
"""
#========================== Uniform Contrast Stimulus =========================
# uniform_contrast = stim.UniformContrast(monitor=mon, indicator=ind, duration=10.,
#                                         color=-1., background=0., pregap_dur=2.,
#                                         postgap_dur=3., coordinate='degree')
# ds = DisplaySequence(log_dir=r'C:\data',
#                      backupdir=None,
#                      display_iter=2,
#                      is_triggered=False,
#                      is_sync_pulse=False,
#                      display_screen=1,
#                      by_index=True)
#
# ds.set_stim(uniform_contrast)
# ds.trigger_display()
#==============================================================================


#======================= Flashing Circle Stimulus =============================
# flashing_circle = stim.FlashingCircle(monitor=mon, indicator=ind, coordinate='degree',
#                                       center=(20., 30.), radius=30., color=-1.,
#                                       flash_frame_num=30, pregap_dur=2.,
#                                       postgap_dur=3., background=0.,
#                                       is_smooth_edge=True, smooth_width_ratio=0.2,
#                                       smooth_func=stim.blur_cos)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, by_index=False, display_iter=2,
#                      display_screen=0)
# ds.set_stim(flashing_circle)
# ds.trigger_display()
#==============================================================================


#======================== Sparse Noise Stimulus ===============================
# sparse_noise = stim.SparseNoise(mon, ind, subregion=(-20.,20.,10.,150.), grid_space=(4., 4.),
#                                 background=0., sign='ON-OFF', pregap_dur=0., postgap_dur=0.,
#                                 coordinate='degree', probe_size=(4., 4.), probe_orientation=0.,
#                                 probe_frame_num=6, iteration=2, is_include_edge = True)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, display_iter=2, display_screen=1,
#                      by_index=True)
# ds.set_stim(sparse_noise)
# ds.trigger_display()
#==============================================================================

#======================= Sparse Noise pt 2 ====================================
# sparse_noise = stim.SparseNoise(mon, ind, subregion=(-30.,10.,30.,90.), grid_space=(8., 8.),
#                                 background=0., sign='ON', pregap_dur=0., postgap_dur=0.,
#                                 coordinate='degree', probe_size=(8., 8.), probe_orientation=0.,
#                                 probe_frame_num=6, iteration=2, is_include_edge=False)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, display_iter=2, display_screen=0,
#                      by_index=False)
# ds.set_stim(sparse_noise)
# ds.trigger_display()
#==============================================================================

#======================= Locally Sparse Noise ====================================
# sparse_noise = stim.LocallySparseNoise(mon, ind, subregion=(-30.,10.,30.,90.), grid_space=(8., 8.),
#                                        background=0., sign='ON-OFF', pregap_dur=0., postgap_dur=0.,
#                                        coordinate='degree', probe_size=(4., 10.), probe_orientation=30.,
#                                        probe_frame_num=8, iteration=2, is_include_edge=True,
#                                        min_distance=50.)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, display_iter=1, display_screen=0,
#                      by_index=True)
# ds.set_stim(sparse_noise)
# ds.trigger_display()
#==============================================================================

#======================= Locally Sparse Noise ====================================
# sparse_noise = stim.LocallySparseNoise(mon, ind, subregion=None, grid_space=(8., 8.),
#                                        background=0., sign='ON-OFF', pregap_dur=0., postgap_dur=0.,
#                                        coordinate='degree', probe_size=(8., 8.), probe_orientation=0.,
#                                        probe_frame_num=15, iteration=10, is_include_edge=True,
#                                        min_distance=50.)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, display_iter=1, display_screen=0,
#                      by_index=True)
# ds.set_stim(sparse_noise)
# ds.trigger_display()
#==============================================================================

#======================= Drifting Grating Circle Stimulus =====================
# dg = stim.DriftingGratingCircle(mon, ind, background=0., coordinate='degree',
#                                 center=(10., 90.), sf_list=(0.02,), tf_list=(1.0,),
#                                 dire_list=(45.,), con_list=(0.8,), radius_list=(20.,),
#                                 block_dur=10., midgap_dur=1., iteration=1, pregap_dur=2.,
#                                 postgap_dur=3., is_smooth_edge=True, smooth_width_ratio=0.2,
#                                 smooth_func=stim.blur_cos)
#
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, display_iter=1, is_triggered=False,
#                      is_sync_pulse=False, is_interpolate=False, display_screen=0,
#                      by_index=False)
#
# ds.set_stim(dg)
# ds.trigger_display()
#==============================================================================

#======================== Drifting Grating pt 2 ===============================
# dg = stim.DriftingGratingCircle(mon, ind, background=0., coordinate='degree',
#                                 center=(-10., 70.), sf_list=[0.02, 0.08],
#                                 tf_list=[4., 1.], dire_list=[30.,], con_list=[1.,0.5],
#                                 radius_list=[10., 30.], block_dur=2., pregap_dur=2.,
#                                 postgap_dur=3., midgap_dur=1., iteration=2,
#                                 is_smooth_edge=True, smooth_width_ratio=0.2,
#                                 smooth_func=stim.blur_cos)
#
# ds=DisplaySequence(log_dir=r'C:\data', backupdir=None, display_iter=1,
#                    is_triggered=False, is_sync_pulse=False, is_interpolate=False,
#                    display_screen=0, by_index=True)
# ds.set_stim(dg)
# ds.trigger_display()
#==============================================================================


#===================== Kalatsky&Stryker Stimulus ==============================
# KS_stim = stim.KSstim(mon,
#                    ind,
#                    coordinate='degree',
#                    sweep_frame=1,
#                    flicker_frame=100)
#
# ds = DisplaySequence(log_dir=r'C:\data',
#                      backupdir=None,
#                      is_triggered=False,
#                      is_sync_pulse=False,
#                      display_iter=2,
#                      display_screen=1)
# ds.set_stim(KS_stim)
# ds.trigger_display()
#==============================================================================

#======================= Kalatsky&Stryker pt 2 ================================
# KS_stim_all_dir = stim.KSstimAllDir(mon,ind,step_width=0.3)
# ds = DisplaySequence(log_dir=r'C:\data',
#                     backupdir=None,
#                     display_iter = 2,
#                     is_triggered=False,
#                     is_sync_pulse=False,
#                     display_screen=1)
# ds.set_stim(KS_stim_all_dir)
# ds.trigger_display()
#==============================================================================

#======================= static grating cricle ================================
# sgc = stim.StaticGratingCircle(monitor=mon, indicator=ind, background=0.,
#                                coordinate='degree', center=(0., 30.), sf_list=(0.02, 0.04, 0.08),
#                                ori_list=(0., 45., 90., 135.), con_list=(0.5, 0.8),
#                                radius_list=(30., 50.), phase_list=(0., 90., 180., 270.),
#                                display_dur=0.25, midgap_dur=0., iteration=2, pregap_dur=2.,
#                                postgap_dur=3., is_smooth_edge=True, smooth_width_ratio=0.2,
#                                smooth_func=stim.blur_cos)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, display_iter=1, display_screen=0,
#                      by_index=True)
# ds.set_stim(sgc)
# ds.trigger_display()
#==============================================================================

#======================= static grating cricle ================================
# sgc = stim.StaticGratingCircle(monitor=mon, indicator=ind, background=0.,
#                                coordinate='degree', center=(0., 30.), sf_list=(0.02,),
#                                ori_list=(0.,), con_list=(0.5,),
#                                radius_list=(20., 50.,), phase_list=(0.,),
#                                display_dur=0.25, midgap_dur=0., iteration=50,
#                                pregap_dur=2., postgap_dur=3.,
#                                is_smooth_edge=True, smooth_width_ratio=0.2,
#                                smooth_func=stim.blur_cos)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, display_iter=1, display_screen=0,
#                      by_index=True, identifier='test')
# ds.set_stim(sgc)
# ds.trigger_display()
#==============================================================================

#======================= stimulus separator ================================
ss = stim.StimulusSeparator(monitor=mon, indicator=ind, coordinate='degree',
                            background=0., indicator_on_frame_num=4,
                            indicator_off_frame_num=4, cycle_num=10,
                            pregap_dur=0., postgap_dur=0.)
ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
                     is_sync_pulse=False, display_iter=1, display_screen=0,
                     by_index=True, identifier='test')
ds.set_stim(ss)
ds.trigger_display()
#==============================================================================