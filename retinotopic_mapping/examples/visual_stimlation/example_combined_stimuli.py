# -*- coding: utf-8 -*-
"""
Example script to test StimulusRoutines.CombinedStimuli class
"""

import numpy as np
import matplotlib.pyplot as plt
import retinotopic_mapping.StimulusRoutines as stim
from retinotopic_mapping.MonitorSetup import Monitor, Indicator
from retinotopic_mapping.DisplayStimulus import DisplaySequence

#============================ monitor setup ======================================
mon_resolution = (1200,1920) #enter your monitors resolution
mon_width_cm = 52 #enter your monitors width in cm
mon_height_cm = 32 #enter your monitors height in cm
mon_refresh_rate = 60  #enter your monitors height in Hz
mon_C2T_cm = mon_height_cm / 2.
mon_C2A_cm = mon_width_cm / 2.
mon_tilt = 30.
mon_dis = 15.
mon_downsample_rate = 5
#=================================================================================

#============================ indicator setup ====================================
ind_width_cm = 3.
ind_height_cm = 3.
ind_position = 'northeast'
ind_is_sync = 'True'
ind_freq = 2.
#=================================================================================

#============================ DisplaySequence ====================================
ds_log_dir = r'C:\data'
ds_backupdir = None
ds_identifier = 'TEST'
ds_display_iter = 1
ds_mouse_id = 'MOUSE'
ds_user_id = 'USER'
ds_psychopy_mon = 'testMonitor'
ds_is_by_index = True
ds_is_interpolate = False
ds_is_triggered=False
ds_trigger_event="negative_edge"
ds_trigger_NI_dev='Dev1'
ds_trigger_NI_port=1
ds_trigger_NI_line=0
ds_is_sync_pulse=False
ds_sync_pulse_NI_dev='Dev1'
ds_sync_pulse_NI_port=1
ds_sync_pulse_NI_line=1
ds_display_screen=0
ds_initial_background_color=0.
#=================================================================================

#============================ generic stimulus parameters ========================
pregap_dur = 2.
postgap_dur = 3.
background = 0.
coordinate = 'degree'
#=================================================================================

#============================ UniformContrast ====================================
uc_duration = 10.
uc_color = -1
#=================================================================================

#============================ FlashingCircle =====================================
fc_center = (20., 30.)
fc_radius = 30.
fc_color = -1.
fc_flash_frame_num = 30
fc_is_smooth_edge = True
fc_smooth_width_ratio = 0.2
fc_smooth_func = stim.blur_cos
#=================================================================================

#============================ SparseNoise ========================================
#=================================================================================

#============================ LocallySparseNoise =================================
#=================================================================================

#============================ DriftingGratingCircle ==============================
#=================================================================================

#============================ StaticGratingCirlce ================================
#=================================================================================

#============================ StaticImages =======================================
#=================================================================================

#============================ StimulusSeparator ==================================
#=================================================================================





#================ Initialize the monitor object ==================================
mon = Monitor(resolution=mon_resolution, dis=mon_dis, mon_width_cm=mon_width_cm,
              mon_height_cm=mon_height_cm, C2T_cm=mon_C2T_cm, C2A_cm=mon_C2A_cm,
              mon_tilt=mon_tilt, downsample_rate=mon_downsample_rate)
# mon.plot_map()
# plt.show()
#=================================================================================

#================ Initialize the indicator object ================================
ind = Indicator(mon, width_cm=ind_width_cm, height_cm=ind_height_cm,
                position=ind_position, is_sync=ind_is_sync, freq=ind_freq)
#=================================================================================

#================ Initialize the DisplaySequence object ==========================
ds = DisplaySequence(log_dir=ds_log_dir, backupdir=ds_backupdir,
                     identifier=ds_identifier, display_iter=ds_display_iter,
                     mouse_id=ds_mouse_id, user_id=ds_user_id,
                     psychopy_mon=ds_psychopy_mon, is_by_index=ds_is_by_index,
                     is_interpolate=ds_is_interpolate, is_triggered=ds_is_triggered,
                     trigger_event=ds_trigger_event, trigger_NI_dev=ds_trigger_NI_dev,
                     trigger_NI_port=ds_trigger_NI_port, trigger_NI_line=ds_trigger_NI_line,
                     is_sync_pulse=ds_is_sync_pulse, sync_pulse_NI_dev=ds_sync_pulse_NI_dev,
                     sync_pulse_NI_port=ds_sync_pulse_NI_port,
                     sync_pulse_NI_line=ds_sync_pulse_NI_line,
                     display_screen=ds_display_screen,
                     initial_background_color=ds_initial_background_color)
#=================================================================================


#========================== Uniform Contrast Stimulus =========================
uc = stim.UniformContrast(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                          postgap_dur=postgap_dur, coordinate=coordinate,
                          background=background, duration=uc_duration,
                          color=uc_color)
#==============================================================================


#======================= Flashing Circle Stimulus =============================
fc = stim.FlashingCircle(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                         postgap_dur=postgap_dur, coordinate=coordinate,
                         background=background, center=fc_center, radius=fc_radius,
                         color=fc_color, flash_frame_num=fc_flash_frame_num,
                         is_smooth_edge=fc_is_smooth_edge,
                         smooth_width_ratio=fc_smooth_width_ratio,
                         smooth_func=fc_smooth_func)
#==============================================================================


#======================== Sparse Noise Stimulus ===============================
# sparse_noise = stim.SparseNoise(mon, ind, subregion=(-20.,20.,10.,150.), grid_space=(4., 4.),
#                                 background=0., sign='ON-OFF', pregap_dur=0., postgap_dur=0.,
#                                 coordinate='degree', probe_size=(4., 4.), probe_orientation=0.,
#                                 probe_frame_num=6, iteration=2, is_include_edge = True)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, display_iter=2, display_screen=1,
#                      is_by_index=True)
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
#                      is_by_index=False)
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
#                      is_by_index=True)
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
#                      is_by_index=True)
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
#                      is_by_index=False)
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
#                    display_screen=0, is_by_index=True)
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
#                      is_by_index=True)
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
#                      is_by_index=True, identifier='test')
# ds.set_stim(sgc)
# ds.trigger_display()
#==============================================================================

#======================= stimulus separator ================================
# ss = stim.StimulusSeparator(monitor=mon, indicator=ind, coordinate='degree',
#                             background=0., indicator_on_frame_num=4,
#                             indicator_off_frame_num=4, cycle_num=10,
#                             pregap_dur=0., postgap_dur=0.)
# ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
#                      is_sync_pulse=False, display_iter=1, display_screen=0,
#                      is_by_index=True, identifier='test')
# ds.set_stim(ss)
# ds.trigger_display()
#==============================================================================

#=============================== static images ================================
si = stim.StaticImages(monitor=mon, indicator=ind, background=0., coordinate='degree',
                       img_center=(0., 60.), deg_per_pixel=(0.1, 0.1), display_dur=0.25,
                       midgap_dur=0.1, iteration=2, pregap_dur=2., postgap_dur=3.)
ds = DisplaySequence(log_dir=r'C:\data', backupdir=None, is_triggered=False,
                     is_sync_pulse=False, display_iter=1, display_screen=0,
                     is_by_index=True, identifier='test')
si.set_imgs_from_hdf5(imgs_file_path=r"D:\data2\rabies_tracing_project\method_development"
                                     r"\2017-09-06-natural-scenes\wrapped_images_for_display.hdf5")
ds.set_stim(si)
ds.trigger_display()
#==============================================================================
