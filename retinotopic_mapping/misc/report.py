# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:20:55 2016

@author: derricw
"""
import sys
import os
import traceback
import tempfile

import numpy as np
import matplotlib.image as mpimg
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as PDFImage, Table as PDFTable
#from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black, red, green, blue

STYLES = getSampleStyleSheet()
QC_STYLES = {
    'Pass': ParagraphStyle('Pass',
                           parent=STYLES['BodyText'],
                           textColor=green),
    'Fail': ParagraphStyle('Fail',
                           parent=STYLES['BodyText'],
                           textColor=red),
    'Review': ParagraphStyle('Review',
                             parent=STYLES['BodyText'],
                             textColor=blue)
}
PASS = QC_STYLES['Pass']
FAIL = QC_STYLES['Fail']
REVIEW = QC_STYLES['Review']


def seconds2hms(seconds):
    """
    Converts seconds to h:mm:ss string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

class Text(object):
    """
    Regular text for our report.
    """
    def __init__(self, string, style=STYLES['BodyText']):
        self.string = string
        self.style = style
        
    def as_plain_text(self):
        return self.string + "\n"
        
    def as_pdf(self):
        return Paragraph(self.string, style=self.style)

class Header(object):
    """
    A header for our report.  Higher values for weight make the font smaller.
    """
    def __init__(self, string, weight=1, style="*"):
        self.string = string
        self.style = style
        self.weight = weight

    def as_plain_text(self):
        return "\n\t*** {} ***\n".format(self.string)

    def as_pdf(self):
        return Paragraph(self.string,
                         style=STYLES['Heading{}'.format(self.weight)])

class Code(object):
    """
    Unused at the moment
    """
    def __init__(self, string):
        self.string = string
        
    def as_plain_text(self):
        return self.string + "\n"
        
    def as_pdf(self):
        return Paragraph(self.string, style=STYLES['Code'])
                                         
class Image(object):
    """
    Insert an image into our report.
    """
    def __init__(self, path):
        self.path = path
        
    def as_plain_text(self):
        return "\n[image](%s)\n" % self.path
        
    def as_pdf(self):
        img = mpimg.imread(self.path)
        h, w, c = img.shape
        width = 4.0
        height = width / w * h
        return PDFImage(self.path,
                        #hAlign='CENTER',
                        width=width*inch,
                        height=height*inch,
                        )
                        
class Table(object):
    """
    Unused at the moment.
    """
    def __init__(self, matrix):
        self.matrix = matrix



class ExperimentReport(object):
    """
    Generic Report.  Has headers, text, images.  Can write as PDF or plain text.
    """
    def __init__(self, experiment):
        self.experiment = experiment

        self.contents = []
        
    def add_text(self, text, style=STYLES['BodyText']):
        self.contents.append(Text(text, style))
    
    def add_header(self, header, weight=1):
        self.contents.append(Header(header, weight))
        
    def add_code(self, code):
        self.contents.append(Code(code))
    
    def add_image(self, image):
        self.contents.append(Image(image))
        
    def to_pdf(self, file_name=""):
        if not file_name:
            # get exp_id if one exists
            exp_id = os.path.basename(self.experiment).split("_")[0] or ""
            file_name = os.path.join(self.experiment, "{}_report.pdf".format(exp_id))
	    self.file_name = file_name
        pdf = PdfDocument(file_name)
        for item in self.contents:
            pdf.add_item(item)
        pdf.write()
            
    def to_plain_text(self, file_name=""):
        if not file_name:
            file_name = os.path.join(self.experiment, "report.txt")
        ptd = PlainTextDocument(file_name)
        for item in self.contents:
            ptd.add_item(item)
        ptd.write()
        


class OphysReport(ExperimentReport):
    """
    Creates a session report for an ophys experiment.
    
    Format is:

     - Intro header with timestamps, rig ID, exp ID
     - Duration of major data streams
     - Stimulus frame info and plots
     - Photodiode / display lag info and plots
     - Video monitoring frame info and plots
     - TwoP frame info and plots
     - Encoder plot
     
    Args:
        experiment_folder (str): the folder containing the sync and pkl data
    
    """
    
    def __init__(self, experiment_folder, other_data_folders=[]):
        super(OphysReport, self).__init__(experiment_folder)
        
        self.passed = True  # experiment passed
        self.review = False  # experiment tentatively passed, should be reviewed
        
        self.load_session(experiment_folder, other_data_folders)

        self._temp_dir = tempfile.gettempdir()
        
        for f, section in [
            (self.build_intro, "Intro"),
            (self.build_duration_info, "Duration"),
            (self.build_stimulus_info, "Stimulus"),
            (self.build_photodiode_info, "Photodiode"),
            (self.build_video_info, "Video Monitoring"),
            (self.build_twop_info, "Two Photon"),
            (self.build_encoder_info, "Encoder"),
        ]:
            try:
                f()
            except:
                tb = traceback.format_exc()
                self.build_failure_report(tb, section)
                
        self.qc()

    def _get_temp_path(self, filename):
        """
        Temporary storage for image plots before they are put into the pdf.
        """
        return os.path.join(self._temp_dir, filename)
        
    def load_session(self, folder, other_data_folders):
        try:
            from ophystools import OphysSession
            self.session = OphysSession(folder, other_data_folders)
            self.session.load_auto()
        except Exception:
            tb = traceback.format_exc()
            self.build_failure_report(tb, "Loading Session")
            
    def build_failure_report(self, tback, section):
        """
        Major failure.  Data file missing or unable to build report segment.

        What all should this do?  Should it email someone?
        """
        failure_str = "Failed to build section '{}' because: \n{}".format(section, tback)
        print(failure_str)
        self.add_text(failure_str, style=FAIL)
        self.passed = False
        
    def build_intro(self):
        """
        Builds the introduction/header.
        """
        exp_id = os.path.split(self.session.folder)[-1].strip("ophys_experiment_")
        title_str = "Ophys Experiment {}".format(exp_id)
        self.add_header(title_str, 1)

        #self.session.load_platform()
        try:
            timestamp = self.session.timestamp
        except KeyError as e:
            timestamp = "failed to get timestamp from platform.json"

        time_str = "\tAcquired: {}".format(timestamp)
        self.add_header(time_str, 4)
        rig_str = "\tRig ID: {}".format(self.session.rig_id)
        self.add_header(rig_str, 4)
            
    def build_duration_info(self):
        """
        Builds duration info segment.
        """
        self.add_header("Duration", 2)
        durations = self.session.duration_info
        self.add_text("TwoPhoton: {}".format(seconds2hms(durations['twop'])))
        self.add_text("Stimulus: {}".format(seconds2hms(durations['stimulus'])))
        self.add_text("Video: {}, {}".format(*[seconds2hms(t) for t in durations['video_monitoring']]))
        

    def build_stimulus_info(self):
        """
        Builds stimulus info.

        - Script name
        - Expected duration
        - Checks vsync count in sync and compares to pkl file
        - Calculates long frames (>0.1s) and dropped frames (>0.025s)
        - Plot frame intervals

        """
        stim_script = self.session.stim_script
        
        stim_vsyncs_pkl = self.session.stim_vsyncs_pkl
        expected_dur_sec = float(stim_vsyncs_pkl)/60.0  # hard coded 60.0?
        expected_dur_str = seconds2hms(expected_dur_sec)
        
        self.add_header("Stimulus: {} - {}".format(stim_script,
                                                   expected_dur_str), 3)
        
        stim_vsyncs_sync = self.session.stim_vsyncs_sync
        if stim_vsyncs_pkl == stim_vsyncs_sync:
            style = PASS
        else:
            style = FAIL
            self.passed = False
        vsyncs_str = "\tStimulus vsyncs: \n\t\t{}(pkl) {}(sync)".format(
            stim_vsyncs_pkl,
            stim_vsyncs_sync,)
        self.add_text(vsyncs_str, style=style)
        
        self.dropped_frames = len(self.session.sync_data.get_long_stim_frames()['intervals'])
        self.long_frames = len(self.session.sync_data.get_long_stim_frames(0.100)['intervals'])
        
        self.add_text("\tDropped frames (>0.25s): {}".format(self.dropped_frames))
        self.add_text("\tLong frames (>0.10s): {}".format(self.long_frames))

        # avg frame interval
        avg_interval = np.mean(self.session.sync_data.get_stim_vsync_intervals())
        self.add_text("Average frame interval: {0:.6f}".format(avg_interval))
        
        # vsync interval plot
        plot_path = self._get_temp_path("frame_intervals.png")
        self.session.sync_data.plot_stim_vsync_intervals(plot_path)
        self.add_image(plot_path)
            
            
    def build_photodiode_info(self):
        """
        Builds photodiode info.

        - Calculate expected vs actual events
        - Calculate photodiode anomalies
        - Calculate display lag
        - Plot start/end of experiment.

        """
        self.add_header("Photodiode", 3)
        
        # Events
        expected = self.session.photodiode_events_pkl
        actual = self.session.photodiode_events_sync
        anomalies = self.session.photodiode_anomalies
        if expected + anomalies == actual:
            style = PASS
        else:
            style = FAIL
            self.review = True
        self.add_text("Photodiode events: {} (expected) {} (actual)".format(expected,
                      actual-anomalies), style=style)
        self.add_text("Photodiode anomalies: {}".format(anomalies))
        
        # Display Lag
        display_lag = self.session.sync_data.display_lag
        if 0 < display_lag < 0.150:
            style = PASS
        else:
            style = FAIL
            self.review = True
        self.add_text("Display lag: {0:.6f}".format(display_lag), style=style)

        # Start/End
        start_path = self._get_temp_path("start.png")
        end_path = self._get_temp_path("end.png")
        self.session.sync_data.plot_start(start_path)
        self.session.sync_data.plot_end(end_path)
        start = self.session.sync_data.stimulus_start
        end = self.session.sync_data.stimulus_end
        self.add_header("Experiment Start: {0:.2f}".format(start), 4)
        self.add_image(start_path)
        self.add_header("Experiment End: {0:.2f}".format(end), 4)
        self.add_image(end_path)
            
    def build_encoder_info(self):
        """
        Builds encoder information.

        - Calculates distance travelled
        - Plots mouse velocity

        """
        self.add_header("Encoder", 3)
        
        # distance travelled?
        dist = self.session.distance_travelled
        self.add_text("Distance travelled (cm): {0:.2f}".format(dist))
        
        # just plot it i guess
        encoder_path = self._get_temp_path("encoder.png")
        self.session.plot_encoder_data(encoder_path)
        self.add_image(encoder_path)
            
    
    def build_video_info(self):
        """
        Builds video monitoring information.

        - Compares frame count in video file, metadata file, and sync file for each video
        - Plots frame intervals for all videos.

        """
        self.add_header("Video Monitoring", 3)
        
        frames_sync = self.session.video_frames_sync
        frames_meta = self.session.video_frames_meta
        frames_avi = self.session.video_frames_avi
        vsyncs_sync = self.session.video_vsyncs
        
        cameras = ["Behavior", "Eyetracking"]
        
        for i, (s, m, a, v, camera) in enumerate(zip(frames_sync,
                                                     frames_meta,
                                                     frames_avi,
                                                     vsyncs_sync,
                                                     cameras)):
            if s==m==a:
                style = PASS
            else:
                style = FAIL
                self.review = True
            self.add_header(camera, 4)
            self.add_text("\tFrames: {} (sync) {} (avi) {} (meta)".format(
                s, a, m), style=style)
            avg_interval = np.mean(np.diff(v))
            self.add_text("Average frame interval: {0:.6f}".format(avg_interval))
        
        vsync_path = self._get_temp_path("video_vsync.png")
        self.session.sync_data.plot_videomon_vsync_intervals(vsync_path)
        self.add_image(vsync_path)
    
    def build_twop_info(self):
        """
        Build two-photon information.

        - Looks for dropped frames, and if found, plots them.
        - Calculates avg vsync interval

        """
        self.add_header("Two Photon Data", 3)
        
        vsyncs = self.session.sync_data.get_twop_vsyncs()
        vsyncs_count = len(vsyncs)
        self.add_text("\tVsyncs: {}".format(vsyncs_count))
        dropped = self.session.sync_data.get_long_twop_frames()['times']
        
        avg_interval = np.mean(np.diff(vsyncs))
        self.add_text("Avg vsync interval: {0:.6f}".format(avg_interval))
        if len(dropped) > 0:
            style = FAIL
            self.review = True
        else:
            style = PASS
            
        self.add_text("\tDropped: {}".format(len(dropped)), style=style)

        if len(dropped) > 5:
            self.add_text("Too many dropped vsyncs to plot.")
        elif 5 > len(dropped) > 0:
            self.add_header("Dropped TwoP Vsyncs", 4)
            for i, d in enumerate(dropped):
                img_path = self._get_temp_path("dropped{}.png".format(i))
                self.session.sync_data.plot_timepoint(d,
                                                      out_file=img_path)
                self.add_text("@ t={0:.3f} seconds".format(d))
                self.add_image(img_path)
                
    def qc(self):
        """
        Check for failures.
        """
        if not self.passed:
            text, style = "FAILED", FAIL
        elif self.review:
            text, style = "REVIEW", REVIEW
        else:
            text, style = "PASSED", PASS
        self.contents.insert(2, Text("Verdict: {}".format(text), style))
            
        
class PdfDocument(object):
    """
    A pdf document.
    """
    def __init__(self, path):
        self.path = path
        
        while os.path.exists(self.path):
            os.remove(self.path)
        
        self.doc = SimpleDocTemplate(self.path,
                                     pagesize=letter,
                                     rightMargin=12,
                                     leftMargin=12,
                                     topMargin=12,
                                     bottomMargin=12,)
        self.contents = []
        self.styles = getSampleStyleSheet()

    def add_item(self, item):
        self.contents.append(item.as_pdf())
        
    def write(self):  
        self.doc.build(self.contents)
        
class PlainTextDocument(object):
    """
    A plain text document.
    """
    def __init__(self, path):
        self.path = path
        self.contents = ""
        
    def add_item(self, item):
        self.contents += item.as_plain_text()
        
    def write(self):
        with open(self.path, 'w') as f:
            f.write(self.contents)
        
if __name__ == "__main__":
    exp = OphysReport(r"C:\Users\derricw\Desktop\ophys_experiment_500964514")
    exp.to_pdf()
    #dset = exp.session.sync_data
    #two_p_r = dset.get_rising_edges("2p_vsync")
    #stim_r = dset.get_rising_edges("stim_vsync")
