# -*- coding: utf-8 -*-
"""
Example script to test StimulusRoutines.CombinedStimuli class
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import retinotopic_mapping.StimulusRoutines as stim
from retinotopic_mapping.MonitorSetup import Monitor, Indicator
from retinotopic_mapping.DisplayStimulus import DisplaySequence

# ============================ monitor setup ======================================
mon_resolution = (1200, 1920)  # enter your monitors resolution
mon_width_cm = 52  # enter your monitors width in cm
mon_height_cm = 32  # enter your monitors height in cm
mon_refresh_rate = 60  # enter your monitors height in Hz
mon_C2T_cm = mon_height_cm / 2.
mon_C2A_cm = mon_width_cm / 2.
mon_tilt = 30.
mon_dis = 15.
mon_downsample_rate = 5
# =================================================================================

# ============================ indicator setup ====================================
ind_width_cm = 3.
ind_height_cm = 3.
ind_position = 'northeast'
ind_is_sync = 'True'
ind_freq = 1.
# =================================================================================

# ============================ DisplaySequence ====================================
ds_log_dir = r'C:\data'
ds_backupdir = None
ds_identifier = 'TEST'
ds_display_iter = 1
ds_mouse_id = 'MOUSE'
ds_user_id = 'USER'
ds_psychopy_mon = 'testMonitor'
ds_is_by_index = True
ds_is_interpolate = False
ds_is_triggered = False
ds_trigger_event = "negative_edge"
ds_trigger_NI_dev = 'Dev1'
ds_trigger_NI_port = 1
ds_trigger_NI_line = 0
ds_is_sync_pulse = False
ds_sync_pulse_NI_dev = 'Dev1'
ds_sync_pulse_NI_port = 1
ds_sync_pulse_NI_line = 1
ds_display_screen = 0
ds_initial_background_color = 0.
# =================================================================================

# ============================ generic stimulus parameters ========================
pregap_dur = 2.
postgap_dur = 3.
background = 0.
coordinate = 'degree'
# =================================================================================

# ============================ UniformContrast ====================================
uc_duration = 30.
uc_color = -1
# =================================================================================

# ============================ FlashingCircle =====================================
fc_center = (20., 30.)
fc_radius = 30.
fc_color = -1.
fc_flash_frame_num = 30
fc_midgap_dur = 5.
fc_iteration = 10
fc_is_smooth_edge = True
fc_smooth_width_ratio = 0.2
fc_smooth_func = stim.blur_cos
# =================================================================================

# ============================ SparseNoise ========================================
sn_subregion = (-40., 60., 30., 90.)
sn_grid_space = (10., 10.)
sn_probe_size = (10., 5.)
sn_probe_orientation = 30.
sn_probe_frame_num = 15
sn_sign = 'ON-OFF'
sn_iteration = 5
sn_is_include_edge = True
# =================================================================================

# ============================ LocallySparseNoise =================================
lsn_subregion = None
lsn_min_distance = 40.
lsn_grid_space = (5., 5.)
lsn_probe_size = (5., 10.)
lsn_probe_orientation = 0.
lsn_probe_frame_num = 15
lsn_sign = 'OFF'
lsn_iteration = 20
lsn_is_include_edge = True
# =================================================================================

# ============================ DriftingGratingCircle ==============================
dgc_center = (10., 90.)
dgc_sf_list = (0.01, 0.04, 0.16)
dgc_tf_list = (0.5, 2., 8.,)
dgc_dire_list = np.arange(0., 360., 90.)
dgc_con_list = (0.8,)
dgc_radius_list = (30.,)
dgc_block_dur = 4.
dgc_midgap_dur = 1.
dgc_iteration = 2
dgc_is_smooth_edge = True
dgc_smooth_width_ratio = 0.2
dgc_smooth_func = stim.blur_cos
# =================================================================================

# ============================ StaticGratingCirlce ================================
sgc_center = (0., 40.)
sgc_sf_list = (0.08,)
sgc_ori_list = (0., 90.)
sgc_con_list = (0.5,)
sgc_radius_list = (25.,)
sgc_phase_list = (0., 90., 180., 270.)
sgc_display_dur = 0.25
sgc_midgap_dur = 0.
sgc_iteration = 30
sgc_is_smooth_edge = True
sgc_smooth_width_ratio = 0.2
sgc_smooth_func = stim.blur_cos
# =================================================================================

# ============================ StaticImages =======================================
si_img_center = (0., 60.)
si_deg_per_pixel = (0.1, 0.1)
si_display_dur = 0.25
si_midgap_dur = 0.
si_iteration = 10
si_images_folder = r"D:\data2\rabies_tracing_project\method_development" \
                   r"\2017-09-06-natural-scenes"
# =================================================================================

# ============================ StimulusSeparator ==================================
ss_indicator_on_frame_num = 4
ss_indicator_off_frame_num = 4
ss_cycle_num = 10
# =================================================================================

# ============================ CombinedStimuli ====================================
cs_stim_ind_sequence = [0, 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 7]
# =================================================================================



# ================ Initialize the monitor object ==================================
mon = Monitor(resolution=mon_resolution, dis=mon_dis, mon_width_cm=mon_width_cm,
              mon_height_cm=mon_height_cm, C2T_cm=mon_C2T_cm, C2A_cm=mon_C2A_cm,
              mon_tilt=mon_tilt, downsample_rate=mon_downsample_rate)
# mon.plot_map()
# plt.show()
# =================================================================================

# ================ Initialize the indicator object ================================
ind = Indicator(mon, width_cm=ind_width_cm, height_cm=ind_height_cm,
                position=ind_position, is_sync=ind_is_sync, freq=ind_freq)
# =================================================================================

# ================ Initialize the DisplaySequence object ==========================
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
# =================================================================================

# ========================== Uniform Contrast =====================================
uc = stim.UniformContrast(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                          postgap_dur=postgap_dur, coordinate=coordinate,
                          background=background, duration=uc_duration,
                          color=uc_color)
# =================================================================================

# ======================= Flashing Circle =========================================
fc = stim.FlashingCircle(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                         postgap_dur=postgap_dur, coordinate=coordinate,
                         background=background, center=fc_center, radius=fc_radius,
                         color=fc_color, flash_frame_num=fc_flash_frame_num,
                         midgap_dur=fc_midgap_dur, iteration=fc_iteration,
                         is_smooth_edge=fc_is_smooth_edge,
                         smooth_width_ratio=fc_smooth_width_ratio,
                         smooth_func=fc_smooth_func)
# =================================================================================

# ======================== Sparse Noise ===========================================
sn = stim.SparseNoise(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                      postgap_dur=postgap_dur, coordinate=coordinate,
                      background=background, subregion=sn_subregion,
                      grid_space=sn_grid_space, sign=sn_sign,
                      probe_size=sn_probe_size, probe_orientation=sn_probe_orientation,
                      probe_frame_num=sn_probe_frame_num, iteration=sn_iteration,
                      is_include_edge=sn_is_include_edge)
# =================================================================================

# ======================= Locally Sparse Noise ====================================
lsn = stim.LocallySparseNoise(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                              postgap_dur=postgap_dur, coordinate=coordinate,
                              background=background, subregion=lsn_subregion,
                              grid_space=lsn_grid_space, sign=lsn_sign,
                              probe_size=lsn_probe_size, probe_orientation=lsn_probe_orientation,
                              probe_frame_num=lsn_probe_frame_num, iteration=lsn_iteration,
                              is_include_edge=lsn_is_include_edge, min_distance=lsn_min_distance)
# =================================================================================

# ======================= Drifting Grating Circle =================================
dgc = stim.DriftingGratingCircle(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                                 postgap_dur=postgap_dur, coordinate=coordinate,
                                 background=background, center=dgc_center,
                                 sf_list=dgc_sf_list, tf_list=dgc_tf_list,
                                 dire_list=dgc_dire_list, con_list=dgc_con_list,
                                 radius_list=dgc_radius_list, block_dur=dgc_block_dur,
                                 midgap_dur=dgc_midgap_dur, iteration=dgc_iteration,
                                 is_smooth_edge=dgc_is_smooth_edge,
                                 smooth_width_ratio=dgc_smooth_width_ratio,
                                 smooth_func=dgc_smooth_func)
# =================================================================================

# ======================= Static Grating Cricle ===================================
sgc = stim.StaticGratingCircle(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                               postgap_dur=postgap_dur, coordinate=coordinate,
                               background=background, center=sgc_center,
                               sf_list=sgc_sf_list, ori_list=sgc_ori_list,
                               con_list=sgc_con_list, radius_list=sgc_radius_list,
                               phase_list=sgc_phase_list, display_dur=sgc_display_dur,
                               midgap_dur=sgc_midgap_dur, iteration=sgc_iteration,
                               is_smooth_edge=sgc_is_smooth_edge,
                               smooth_width_ratio=sgc_smooth_width_ratio,
                               smooth_func=sgc_smooth_func)
# =================================================================================

# =============================== Static Images ===================================
si = stim.StaticImages(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                       postgap_dur=postgap_dur, coordinate=coordinate,
                       background=background, img_center=si_img_center,
                       deg_per_pixel=si_deg_per_pixel, display_dur=si_display_dur,
                       midgap_dur=si_midgap_dur, iteration=si_iteration)
# =================================================================================

# ============================ wrape images =======================================
print ('wrapping images ...')
static_images_path = os.path.join(si_images_folder, 'wrapped_images_for_display.hdf5')
if os.path.isfile(static_images_path):
    os.remove(static_images_path)
si.wrap_images(si_images_folder)
# =================================================================================

# ======================= Stimulus Separator ======================================
ss = stim.StimulusSeparator(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                            postgap_dur=postgap_dur, coordinate=coordinate,
                            background=background,
                            indicator_on_frame_num=ss_indicator_on_frame_num,
                            indicator_off_frame_num=ss_indicator_off_frame_num,
                            cycle_num=ss_cycle_num)
# =================================================================================

# ======================= Combined Stimuli ========================================
cs = stim.CombinedStimuli(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                          postgap_dur=postgap_dur, coordinate=coordinate,
                          background=background)
# =================================================================================

# ======================= Set Stimuli Sequence ====================================
all_stim = [uc, fc, sn, lsn, dgc, sgc, si, ss]
stim_seq = [all_stim[stim_ind] for stim_ind in cs_stim_ind_sequence]
cs.set_stimuli(stimuli=stim_seq, static_images_path=static_images_path)
# =================================================================================

# =============================== display =========================================
ds.set_stim(cs)
ds.trigger_display()
plt.show()
# =================================================================================
