import mfreeze.freezelib
import mfreeze.locolib
import mfreeze.video
import numpy as np
import mfreeze.utils
from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from functools import wraps


def _update_plot_interactive(f):
    """
    Update crop attrs before running func
    """

    @wraps(f)
    def wrapped(self, *args, **kwargs):
        if self._using_crop_interactive:
            self._update_crop_attrs_interactive()
        return f(self, *args, **kwargs)

    return wrapped


def _run_check(f):
    """
    Raise Error if trying to perform f before the detector has ran
    """

    @wraps(f)
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

    def _update_crop_attrs_interactive(self):
        self.rmin, self.rmax, self.cmin, self.cmax = mfreeze.utils._unpack_hv_crop(
            self._crop_obj
        )

    def interactive_crop(self, frame=0):
        """
        Use an interactive cropping tool to crop a subregion of interest. All following analyses
        will use this subrgion.

        Designed to run in a jupyter notebook. To display the image, print it. To use the boxcrop, click
        the icon on the lower right of the image and double click image to start cropping.

        Returns:
            image, crop_object
        """
        image, boxcrop = mfreeze.video.interactive_crop(self.video_path, frame=frame)
        self._using_crop_interactive = True
        self._crop_obj = boxcrop
        return image, boxcrop

    @_update_plot_interactive
    def get_used_crop_coords(self,):
        return {
            "cmin": self.cmin,
            "cmax": self.cmax,
            "rmin": self.rmin,
            "rmax": self.rmax,
        }


class FreezeDetector(BaseProcessor):
    """
    Freeze Detection.

    A freeze detector finds the frames in a video where an animal was freezing.

    Args:
        video_path (str): Path to the video
        start_frame (int): The first frame to use.
        stop_frame (int): The final frame to use. Defaults to the last frame.
        use_med_filter (bool): Whether to apply a median filter to the motion estimates.
        med_filter_size (int): If using a median filter, the size of the filter to use.
        freeze_threshold (float): The threshold for Freezing. Lower thresholds yeild more frames being classified
                                  as freezes.
        min_freeze_duration (int): Defines the minimum time period for a freeze. Freezes below this threshold will not
                                   be counted as freezes.
        save_video_prefix (str): The string to append to the video name when saving the freeze video.
        save_video_dir (str): The path to the directory to which the freeze video will be saved.
        crop_interactive (holoviews.streams.BoxEdit): Holoviews stream object to use if using the interactive cropping
                                                      functionality. This can be obtained from the mfreeze.video.crop
                                                      function.
        crop_cmin (int): Optional value for manual cropping. Specifies minimum column value for each frame.
        crop_cmax (int): Optional value for manual cropping. Specifies maximum column value for each frame.
        crop_rmin (int): Optional value for manual cropping. Specifies minimum row value for each frame.
        crop_rmax (int): Optional value for manual cropping. Specifies maximum row values for each frame.
    """

    def __init__(
        self,
        video_path,
        start_frame=0,
        stop_frame=None,
        use_med_filter=True,
        motion_algo="MOG",
        med_filter_size=3,
        freeze_threshold=None,
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
        if motion_algo == "MOG":
            self.motion_algo = mfreeze.freezelib.detect_motion_MOG
        elif motion_algo == "ABSDIFF":
            self.motion_algo = mfreeze.freezelib.detect_motion
        else:
            raise ValueError(
                "Unknown Motionion detection algorythm. Select from {'MOG', 'ABSDIFF'}"
            )
        self.use_med_filter = use_med_filter
        self.med_filter_size = med_filter_size
        self.freeze_threshold = freeze_threshold
        self.min_freeze_duration = min_freeze_duration
        self.save_video_prefix = save_video_prefix

    @_update_plot_interactive
    def detect_motion(self, use_med_filter=None, med_filter_size=None):
        """
        TODO freeze documentation
        """
        use_med_filter = (
            use_med_filter if use_med_filter is not None else self.use_med_filter
        )
        med_filter_size = (
            med_filter_size if med_filter_size is not None else self.med_filter_size
        )
        self.motion_ = self.motion_algo(
            self.video_path,
            start_frame=self.start_frame,
            stop_frame=self.stop_frame,
            use_med_filter=self.use_med_filter,
            med_filter_size=self.med_filter_size,
            crop_rmin=self.rmin,
            crop_rmax=self.rmax,
            crop_cmin=self.cmin,
            crop_cmax=self.cmax,
        )
        self._has_ran = True
        return self.motion_

    @_update_plot_interactive
    @_run_check
    def detect_freezes(self, freeze_threshold=None, min_freeze_duration=None):
        """
        TODO: Freeze documentation
        """
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
        """
        TODO: freeze docs
        """
        _, ax = plt.subplots(figsize=figsize)
        ax.plot(self.motion_)
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

    @_run_check
    @_update_plot_interactive
    def save_video(self, outpath=None):
        """
        TODO Freeze docs
        """
        if outpath is None:
            fn = f"{self.save_video_prefix}{self.video_name}"
            outpath = str(Path(self.save_video_dir) / fn)
        self.outpath = outpath
        mfreeze.video.save_freeze_video(
            self.video_path,
            outfile_path=self.outpath,
            freezes=self.freezes_,
            start_frame=self.start_frame,
            stop_frame=self.stop_frame,
            crop_rmin=self.rmin,
            crop_rmax=self.rmax,
            crop_cmin=self.cmin,
            crop_cmax=self.cmax,
        )
        return self

    def run_analysis(self):
        """
        TODO: freeze docs
        """
        self.detect_motion()
        self.detect_freezes()
        self.save_video()
        return self

    @_run_check
    def generate_report(self, include_threshold=True):
        """
        TODO: Freeze docs
        """
        video_name = self.video_name
        fps = self.video_fps
        frame = np.arange(self.start_frame, self.start_frame + len(self.motion_))
        time = np.round(frame * (1 / self.video_fps), 6)
        was_freezing = self.freezes_
        return pd.DataFrame(
            {
                "frame": frame,
                "time": time,
                "video_name": video_name,
                "fps": fps,
                "was_freezing": was_freezing,
            }
        )

    def __repr__(self):
        return f"<FreezeDetector: {self.video_name}>"


class LoctionTracker(BaseProcessor):
    """
    Location Tracker.

    Tracks the frame by frame (x, y) pixel position of the largest moving object in a video.
    
    Args:
        video_path (str): Path to the video
        reference_num_frames (int): The number of frames to use when generating the reference.
        thresh (float): A value between 0 and 100. Change this value if the location parameter is performing poorly.
        start_frame (int): The first frame to use.
        stop_frame (int): The final frame to use. Defaults to the last frame.
        save_video_prefix (str): The string to append to the video name when saving the freeze video.
        save_video_dir (str): The path to the directory to which the freeze video will be saved.
        crop_interactive (holoviews.streams.BoxEdit): Holoviews stream object to use if using the interactive cropping
                                                      functionality. This can be obtained from the mfreeze.video.crop
                                                      function.
        crop_cmin (int): Optional value for manual cropping. Specifies minimum column value for each frame.
        crop_cmax (int): Optional value for manual cropping. Specifies maximum column value for each frame.
        crop_rmin (int): Optional value for manual cropping. Specifies minimum row value for each frame.
        crop_rmax (int): Optional value for manual cropping. Specifies maximum row values for each frame.
    """

    def __init__(
        self,
        video_path,
        reference_num_frames=200,
        thresh=99,
        start_frame=0,
        stop_frame=None,
        save_video_prefix="location_video_",
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
        self.save_video_prefix = save_video_prefix
        self.reference_num_frames = reference_num_frames
        self.thresh = thresh

    @_update_plot_interactive
    def track_location(self, thresh=None):
        """
        Track the location of the largest object in the frame.

        Args:
            thresh (float): A value between 0 and 100. Change this value if the location parameter is performing poorly.
        """
        if thresh is not None:
            self.thresh = thresh

        self.ref_frame_ = mfreeze.locolib.reference_create(
            self.video_path,
            num_frames=self.reference_num_frames,
            start_frame=self.start_frame,
            stop_frame=self.stop_frame,
            crop_cmin=self.cmin,
            crop_cmax=self.cmax,
            crop_rmin=self.rmin,
            crop_rmax=self.rmax,
        )
        self.rc_ = mfreeze.locolib.track_location(
            self.video_path,
            ref_frame=self.ref_frame_,
            thresh=self.thresh,
            start_frame=self.start_frame,
            stop_frame=self.stop_frame,
            crop_cmin=self.cmin,
            crop_cmax=self.cmax,
            crop_rmin=self.rmin,
            crop_rmax=self.rmax,
        )
        self.x_, self.y_ = self.rc_[:, 1].flatten(), self.rc_[:, 0].flatten()
        self._has_ran = True
        return self

    @_run_check
    @_update_plot_interactive
    def save_video(self, outpath=None):
        """
        TODO: LOCO DOCS
        """
        if outpath is None:
            fn = f"{self.save_video_prefix}{self.video_name}"
            outpath = str(Path(self.save_video_dir) / fn)
        self.outpath = outpath
        mfreeze.video.save_loco_video(
            self.video_path,
            outfile_path=self.outpath,
            rc=self.rc_,
            start_frame=self.start_frame,
            stop_frame=self.stop_frame,
            crop_rmin=self.rmin,
            crop_rmax=self.rmax,
            crop_cmin=self.cmin,
            crop_cmax=self.cmax,
        )
        return self

    @_run_check
    def generate_report(self):
        """
        TODO: loco docs
        """
        video_name = self.video_name
        frame = np.arange(self.start_frame, self.start_frame + len(self.x_))
        time = np.round(frame * (1 / self.video_fps), 6)
        fps = self.video_fps
        x = self.x_
        y = self.y_
        return (
            pd.DataFrame(
                {
                    "frame": frame,
                    "time": time,
                    "video_name": video_name,
                    "fps": fps,
                    "x": x,
                    "y": y,
                }
            )
            .assign(
                distance=lambda x: self._distance(
                    x["x"], x.shift()["x"], x["y"], x.shift()["y"]
                )
            )
            .iloc[:-1, :]
        )

    @staticmethod
    def _distance(x_old, x_new, y_old, y_new):
        return np.sqrt((x_old - x_new) ** 2) + ((y_old - y_new) ** 2)

