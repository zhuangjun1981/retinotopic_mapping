'''
2017-10-31 by Jun Zhuang
this module provides analysis tools to extract information about visual stimuli
saved in the log pkl files.
'''

import numpy as np
import tools.FileTools as ft

class DisplayLogAnalyzer(object):
    """
    class to take display_log (.pkl) file, check its integrity and extract stimuli and display
    organize into stim_dict dictionary, which is a intermediate step to put visual display
    information into nwb files.
    """

    def __init__(self, log_path):

        self.log_dict = ft.loadFile(log_path)

        if not self.log_dict['presentation']['is_by_index']:
            raise NotImplementedError('The visual stimuli display should be indexed.')

        self.check_integrity()

    def check_integrity(self):

        print(self.log_dict['presentation']['frame_stats'])

        if not self.log_dict['presentation']['keep_display']:
            raise ValueError('Stimulus presentation did not end normally.')

        total_frame1 = len(self.log_dict['presentation']['displayed_frames'])
        total_frame2 = len(self.log_dict['presentation']['frame_ts_start'])
        total_frame3 = len(self.log_dict['presentation']['frame_ts_end'])
        total_frame4 = len(self.log_dict['stimulation']['index_to_display'])
        if not total_frame1 == total_frame2 == total_frame3 == total_frame4:
            print('\nNumber of displayed frames: {}.'.format(total_frame1))
            print('\nNumber of frame start timestamps: {}.'.format(total_frame2))
            print('\nNumber of frame end timestamps: {}.'.format(total_frame3))
            print('\nNumber of frames to be displayed: {}.'.format(total_frame4))
            raise ValueError('Numbers of total frames do not agree with each other from various places.')

        if max(self.log_dict['stimulation']['index_to_display']) >= \
                len(self.log_dict['stimulation']['frames_unique']):
            raise ValueError('Display index beyond number of unique frames.')

        if self.log_dict['stimulation']['stim_name'] == 'CombinedStimuli':
            stimuli_sequence_out = [f[0] for f in self.log_dict['presentation']['displayed_frames']]
            stimuli_sequence_out = list(set(stimuli_sequence_out))
            stimuli_sequence_out.sort()
            stimuli_sequence_in = self.log_dict['stimulation']['individual_logs'].keys()
            stimuli_sequence_in.sort()
            if stimuli_sequence_out != stimuli_sequence_in:
                raise ValueError('Output stimuli sequence does not match input stimuli sequence.')

    @property
    def num_frame_tot(self):
        return len(self.log_dict['presentation']['displayed_frames'])

    def get_stim_dict(self):
        """
        Returns
        -------
        stim_dict: dictionary
            the structure of this dictionary should look like this:

            {
             '000_UniformContrastRetinotopicMapping': {
                                                       ...
                                                       'stim_name' : '000_UniformContrastRetinotopicMapping',
                                                       'index_to_display': <index referencing 'frames_unique' field>
                                                       'timestamps': <index referencing entire display sequence,
                                                                     should match hardware vsync signal>
                                                       'frames_unique': list of tuple representing unique frames
                                                       ...
                                                       },
             '001_StimulusSeparatorRetinotopicMapping: {
                                                        ...
                                                        'stim_name' : '000_UniformContrastRetinotopicMapping',
                                                        'index_to_display': <index referencing 'frames_unique' field>
                                                        'timestamps': <index referencing entire display sequence,
                                                                       should match hardware vsync signal>
                                                        'frames_unique': list of tuple representing unique frames
                                                        ...
                                                        },
             ...
             }
        """

        comments = ''
        description = ''
        source = 'retinotopic_mapping package'
        stim_dict = {}

        # if multiple stimuli were displayed in a sequence
        if self.log_dict['stimulation']['stim_name'] == 'CombinedStimuli':
            curr_frame_ind = 0
            stim_ids = self.log_dict['stimulation']['individual_logs'].keys()
            stim_ids.sort()
            for stim_id in stim_ids:
                curr_dict = self.log_dict['stimulation']['individual_logs'][stim_id]
                curr_stim_name = stim_id + 'RetinotopicMapping'
                curr_dict['stim_name'] = curr_stim_name
                curr_num_frames = len(curr_dict['index_to_display'])
                curr_dict.update({'timestamps': np.arange(curr_num_frames, dtype=np.uint64) + curr_frame_ind,
                                  'comments': comments,
                                  'source': source,
                                  'description': description})
                curr_frame_ind = curr_frame_ind + curr_num_frames
                stim_dict.update({curr_stim_name: curr_dict})

        # if only one stimulus was displayed
        else:
            stim_name = self.log_dict['stimulation']['stim_name']
            if stim_name in ['UniformContrast', 'FlashingCircle', 'SparseNoise', 'LocallySparseNoise',
                             'DriftingGratingCirlce', 'StaticGratingCircle', 'StaticImages', 'StimulusSeparator']:
                curr_stim_name = '{:03d}_{}RetinotopicMapping'.format(0, stim_name)
                curr_dict = self.log_dict['stimulation']
                curr_dict['stim_name'] = curr_stim_name
                curr_dict.update({'timestamps': np.arange(self.num_frame_tot, dtype=np.uint64)})
            else:
                raise NotImplementedError('Do not understand stimulus: {}.'.format(stim_name))

            curr_dict.update({'comments': comments,
                              'source': source,
                              'description': description})
            stim_dict.update({curr_stim_name : curr_dict})

        return stim_dict




