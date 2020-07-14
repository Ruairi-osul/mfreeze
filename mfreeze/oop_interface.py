import mfreeze.freezelib
import mfreeze.video
import numpy as np
import mfreeze.utils
from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def _update_plot_interactive(f):
    """
    Update crop attrs before
    """

    def wrapped(self, *args, **kwargs):
        if self._using_crop_interactive:
            self._update_crop_attrs_interactive()
        return f(self, *args, **kwargs)

    return wrapped


def _run_check(f):
    """
    Raise Error if trying to perform f before the detector has ran
    """

    def wrapped(self, *args, **kwargs):
        if not self._has_ran:
            raise ValueError("Cannot run this function until the analysis has ran.")
        return f(self, *args, **kwargs)

    return wrapped


class BaseProcessor:
    def __init__(
        self,
        video_path,
        start_frame=0,
        stop_frame=None,
        crop_cmin=None,
        crop_cmax=None,
        crop_rmin=None,
        crop_rmax=None,
        crop_interactive=None,
    ):
        self.video_name = Path(video_path).name
        self.video_path = video_path
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self._has_ran = False
        self.video_fps = self._get_fps()

        # parse crop settings
        nrow, ncol = self._get_frame_size()
        _, crop_settings = mfreeze.utils._parse_crop_settings(
            nrow, ncol, crop_interactive, crop_cmin, crop_cmax, crop_rmin, crop_rmax
        )
        self.rmin, self.rmax, self.cmin, self.cmax = crop_settings
        self._using_crop_interactive = True if crop_interactive is not None else False

    def _get_frame_size(self):
        cap = cv2.VideoCapture(self.video_path)
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nrow, ncol = frame.shape
        cap.release()
        return nrow, ncol

    def _get_fps(self):
        cap = cv2.VideoCapture(self.video_path)
        return cap.get(cv2.CAP_PROP_FPS)

    def interactive_crop(self):
        image, boxcrop = mfreeze.video.interactive_crop(
            self.video_path, frame=self.start_frame
        )
        self._using_crop_interactive = True
        self._crop_obj = boxcrop
        return image, boxcrop

    def _update_crop_attrs_interactive(self):
        self.rmin, self.rmax, self.cmin, self.cmax = mfreeze.utils._unpack_hv_crop(
            self._crop_obj
        )

    @_update_plot_interactive
    def get_used_crop_coords(self,):
        return {
            "cmin": self.cmin,
            "cmax": self.cmax,
            "rmin": self.rmin,
            "rmax": self.rmax,
        }


class FreezeDetector(BaseProcessor):
    def __init__(
        self,
        video_path,
        start_frame=0,
        stop_frame=None,
        use_med_filter=True,
        med_filter_size=3,
        freeze_threshold=0.5,
        min_freeze_duration=5,
        save_video_prefix="freeze_video_",
        save_video_dir=None,
        crop_interactive=None,
        crop_cmin=None,
        crop_cmax=None,
        crop_rmin=None,
        crop_rmax=None,
    ):
        super().__init__(
            video_path=video_path,
            start_frame=start_frame,
            stop_frame=stop_frame,
            crop_cmin=crop_cmin,
            crop_cmax=crop_cmax,
            crop_rmin=crop_rmin,
            crop_rmax=crop_rmax,
        )
        self.save_video_dir = (
            Path(".").absolute() if save_video_dir is None else save_video_dir
        )
        self.use_med_filter = use_med_filter
        self.med_filter_size = med_filter_size
        self.freeze_threshold = freeze_threshold
        self.min_freeze_duration = min_freeze_duration
        self.save_video_prefix = save_video_prefix

    @_update_plot_interactive
    def detect_motion(self, use_med_filter=None, med_filter_size=None):
        use_med_filter = (
            use_med_filter if use_med_filter is not None else self.use_med_filter
        )
        med_filter_size = (
            med_filter_size if med_filter_size is not None else self.med_filter_size
        )
        self.motion_ = mfreeze.freezelib.detect_motion(
            self.video_path,
            self.start_frame,
            self.stop_frame,
            use_med_filter=self.use_med_filter,
            med_filter_size=self.med_filter_size,
        )
        self._has_ran = True
        return self.motion_

    @_update_plot_interactive
    @_run_check
    def detect_freezes(self, freeze_threshold=None, min_freeze_duration=None):
        self.freeze_threshold = (
            freeze_threshold if freeze_threshold is not None else self.freeze_threshold
        )
        self.min_freeze_duration = (
            min_freeze_duration
            if min_freeze_duration is not None
            else self.min_freeze_duration
        )
        self.freezes_ = mfreeze.freezelib.detect_freezes(
            self.motion_,
            freeze_threshold=self.freeze_threshold,
            min_duration=self.min_freeze_duration,
        )

    @_run_check
    def plot_motion(self, figsize=(10, 5)):
        _, ax = plt.subplots(figsize=figsize)
        ax.plot(self.freezes_)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalised Motion")
        if self.freeze_threshold is not None:
            ax.axhline(
                self.freeze_threshold,
                linestyle="--",
                color="red",
                label="Current Freeze Threshold",
            )
            plt.legend()
        return ax

    def save_video(self, outpath=None, start_frame=0, stop_frame=None):
        if outpath is None:
            fn = f"{self.save_video_prefix}{self.video_name}"
            outpath = str(self.save_video_dir / fn)
        mfreeze.video.save_video(
            self.video_path,
            outpath,
            self.freezes_,
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
        return self

    def run_analysis(self):
        self.detect_motion()
        self.detect_freezes()
        self.save_video()
        return self

    @_run_check
    def generate_report(self, include_threshold=True):
        video_name = self.video_name
        frame = np.arange(len(self.motion_))
        time = np.round(frame * (1 / self.video_fps), 6)
        freeze_threshold_used = self.freeze_threshold
        minimum_freeze_duration = self.min_freeze_duration
        motion_zscore = self.motion_
        was_freezing = self.freezes_
        return pd.DataFrame(
            {
                "frame": frame,
                "time": time,
                "video_name": video_name,
                "freeze_threshold_used": freeze_threshold_used,
                "minimum_freeze_duration": minimum_freeze_duration,
                "motion_zscore": motion_zscore,
                "was_freezing": was_freezing,
            }
        )

    def __repr__(self):
        return f"<FreezeDetector: {self.video_name}>"


class FreezeDetectorBath:
    def __init__(
        self,
        motion_threshold=None,
        freeze_threshold=None,
        min_freeze_duration=None,
        save_video=False,
        save_video_path=None,
        crop_max_height=None,
        crop_min_height=None,
        crop_max_width=None,
        crop_min_width=None,
    ):
        self._detectors = []
        pass

    def run_analysis(self, video_paths):
        for path in video_paths:
            detector = FreezeDetector().run_analysis(path)
            self._detectors.append(detector)

    @_run_check
    def generate_report(self):
        pass

    @_run_check
    def save_videos(self, save_dir, prefix=None, suffix=None):
        pass
