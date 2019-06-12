import os
import unittest
import retinotopic_mapping.DisplayStimulus as ds

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestSimulation(unittest.TestCase):

    def setUp(self):

        import retinotopic_mapping.MonitorSetup as ms

        # Setup monitor/indicator objects
        self.monitor = ms.Monitor(resolution=(1200, 1600), dis=15.,
                                  mon_width_cm=40., mon_height_cm=30.,
                                  C2T_cm=15., C2A_cm=20., center_coordinates=(0., 60.),
                                  downsample_rate=10)
        # import matplotlib.pyplot as plt
        # self.monitor.plot_map()
        # plt.show()

        self.indicator = ms.Indicator(self.monitor, width_cm=3., height_cm=3., position='northeast',
                                      is_sync=True, freq=1.)

    def test_initial_background(self):

        import retinotopic_mapping.StimulusRoutines as stim

        log_dir = os.path.join(curr_folder, 'test_data')

        displayer = ds.DisplaySequence(log_dir=log_dir, backupdir=None, identifier='TEST', display_iter=1,
                                       mouse_id='MOUSE', user_id='USER', psychopy_mon='testMonitor',
                                       is_by_index=True, is_interpolate=False, is_triggered=False,
                                       is_save_sequence=False, trigger_event="negative_edge",
                                       trigger_NI_dev='Dev1', trigger_NI_port=1, trigger_NI_line=0,
                                       is_sync_pulse=False, sync_pulse_NI_dev='Dev1', sync_pulse_NI_port=1,
                                       sync_pulse_NI_line=1, display_screen=0, initial_background_color=0.,
                                       color_weights=(0., 1., 1.))

        # print(displayer.initial_background_color)

        uc = stim.UniformContrast(monitor=self.monitor, indicator=self.indicator, pregap_dur=0.1,
                                  postgap_dur=0.1, coordinate='degree',
                                  background=0., duration=0.1, color=0.8)

        displayer.set_stim(uc)
        log_path = displayer.trigger_display()

        import shutil
        log_dir = os.path.join(curr_folder, 'test_data', 'visual_display_log')
        shutil.rmtree(log_dir)

