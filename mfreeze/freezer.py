import mfreeze.eztrack_freeze_functions as eztrack_freeze
import numpy as np
import pandas as pd
import holoviews as hv


class FreezeProcessor:
    def __init__(self, video_dict: dict = None):
        self._video_dict = video_dict

    def load_and_crop(
        self, frame: int = 0, width: float = 1, height: float = 1, cropmethod="Box"
    ):
        self._image, self._crop, self._video_dict = eztrack_freeze.LoadAndCrop(
            self._video_dict,
            dict(width=width, height=height),
            cropmethod=cropmethod,
            start_frame=frame,
        )
        return self._image, self._crop, self._video_dict

    def calibrate(
        self,
        num_secs: int = 20,
        start_frame: int = 0,
        cal_pix: int = 10000,
        sigma: int = 1,
    ):
        self._video_dict["cal_sec"] = num_secs
        return eztrack_freeze.Calibrate(
            self._video_dict, cal_pix=cal_pix, SIGMA=sigma, start_frame=start_frame
        )

    def detect_motion(
        self, mt_cutoff: int = 10, sigma: int = 1,
    ):
        """
        Detect motion.
        Args:
            mt_cutoff [int]: Minimum pixel change for motion classification
            sigma [int]: Gausian filter hypterparameter
        Returns:
            motion array [np.ndarray]
        """
        self._mt_cutoff = mt_cutoff
        self._motion = eztrack_freeze.Measure_Motion(
            self._video_dict, mt_cutoff, crop=self._crop, SIGMA=sigma
        )
        return self._motion

    def detect_freezes(
        self, freeze_threshold: int = 150, min_consecutive_frames: int = None
    ):
        """
        Detect freezing events.
        Args:
            freeze_threshold [int]: frames having less than this number of pixels
                                    with motion are classified as freezes
            min_consecurtive_frames [int]: only sets of consequitive frames longer
                                           than this number are classified as
                                           freezes
        """
        if min_consecutive_frames is None:
            min_consecutive_frames = int(self._video_dict.get("fps", 30)) // 3
        self._freeze_threshold = freeze_threshold
        self._min_consecutive_frames = min_consecutive_frames
        self._freezing = eztrack_freeze.Measure_Freezing(
            self._motion, self._freeze_threshold, self._min_consecutive_frames
        )
        return self._freezing

    def plot_motion(self, height: int = 300, width: int = 1000):
        return hv.Curve((np.arange(len(self._motion)), self._motion)).opts(
            height=height,
            width=width,
            line_width=1,
            color="steelblue",
            title="Motion Across Session",
        )

    def plot_freezing(self):
        return hv.Area(
            self._freezing * (self._motion.max() / 100), "Frame", "Motion"
        ).opts(color="lightgray", line_width=0, line_alpha=0)

    def plot_freezing_and_motion(self, height: int = 300, width: int = 1000):
        f_plot = self.plot_freezing()
        m_plot = self.plot_motion()
        return f_plot * m_plot

    def get_freezes(self):
        return self._freezing

    def get_motion(self):
        return self._motion

    def set_save_dir(self, dirname):
        self.save_dir = dirname

    def set_output_video_name(self, vid_name):
        self.set_output_video_name = vid_name

    def get_name(self):
        return self._video_dict.get("file", "vid")

    def get_report(self):
        return (
            pd.DataFrame(
                {
                    "file": self.get_name(),
                    "freezing": np.where(self.get_freezes() == 0, 0, 1),
                    "motion": self.get_motion(),
                }
            )
            .reset_index()
            .rename(columns={"index": "frame"})
        )

    def play_video(
        self,
        start_frame: int = 0,
        end_frame: int = 500,
        save: bool = True,
        fname: str = None,
        save_dir: str = None,
        sigma: float = 1,
    ):
        display_dict = {
            "start": start_frame,
            "end": end_frame,
            "resize": None,
            "fps": self._video_dict.get("fps"),
            "save_video": save,
        }
        if not save_dir:
            try:
                save_dir = self.save_dir
            except AttributeError:
                save_dir = None
        if not fname:
            try:
                fname = self.set_output_video_name
            except AttributeError:
                fname = None

        return eztrack_freeze.PlayVideo(
            self._video_dict,
            display_dict,
            self._freezing,
            self._mt_cutoff,
            crop=self._crop,
            SIGMA=sigma,
            fname=fname,
            save_dir=save_dir,
        )
