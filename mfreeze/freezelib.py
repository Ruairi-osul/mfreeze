import cv2
import numpy as np
from scipy.signal import medfilt
from .utils import _crop_frame, _runs, _parse_crop_settings


def detect_motion(
    video_path,
    use_med_filter=True,
    med_filter_size=3,
    start_frame=0,
    stop_frame=None,
    crop_interactive=None,
    crop_cmin=None,
    crop_cmax=None,
    crop_rmin=None,
    crop_rmax=None,
    compression_factor=None,
):
    """
    Estimates the amount of motion in each frame in of a video.

    The estimation is made by comparing grayscale pixel values in concecutive frames.

    Args:
        video_path (str): Path to the video
        use_med_filter (bool): Whether to apply a median filter to the motion estimates.
        med_filter_size (int): If using a median filter, the size of the filter to use.
        start_frame (int): The first frame to use.
        stop_frame (int): The final frame to use. Defaults to the last frame.
        crop_interactive (holoviews.streams.BoxEdit): Holoviews stream object to use if using the interactive cropping
                                                      functionality. This can be obtained from the mfreeze.video.crop
                                                      function.
        crop_cmin (int): Optional value for manual cropping. Specifies minimum column value for each frame.
        crop_cmax (int): Optional value for manual cropping. Specifies maximum column value for each frame.
        crop_rmin (int): Optional value for manual cropping. Specifies minimum row value for each frame.
        crop_rmax (int): Optional value for manual cropping. Specifies maximum row values for each frame.
    Returns:
        A numpy array containing the number of pixels exceding the motion threshold per frame.
    """
    cap = cv2.VideoCapture(video_path)
    if stop_frame is None:
        stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    _, frame_old = cap.read()
    frame_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
    frame_nrow, frame_ncol = frame_old.shape
    using_crop, crop_settings = _parse_crop_settings(
        frame_nrow,
        frame_ncol,
        crop_interactive,
        crop_cmin,
        crop_cmax,
        crop_rmin,
        crop_rmax,
    )
    if using_crop:
        frame_old = _crop_frame(frame_old, *crop_settings)
    num_frames = stop_frame - start_frame
    motion = np.zeros(num_frames - 1, dtype="uint32")

    for i in range(1, len(motion) + 1):
        frame_present, frame = cap.read()
        if frame_present:
            if using_crop:
                frame = _crop_frame(frame, *crop_settings)
            if compression_factor:
                frame = cv2.resize(
                    frame,
                    dsize=(
                        int(frame.shape[1] / compression_factor),
                        int(frame.shape[0] / compression_factor),
                    ),
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion[i - 1] = np.sum(cv2.absdiff(frame, frame_old))
            frame_old = frame
        else:
            i -= 2  # Reset x to last frame detected
            motion = motion[:i]
            break
    cap.release()
    motion = (motion - np.mean(motion)) / np.std(motion)
    if use_med_filter:
        motion = medfilt(motion, med_filter_size)
    return motion


def detect_motion_MOG(
    video_path,
    use_med_filter=True,
    med_filter_size=3,
    start_frame=0,
    stop_frame=None,
    crop_interactive=None,
    MOG_history=300,
    crop_cmin=None,
    crop_cmax=None,
    crop_rmin=None,
    crop_rmax=None,
    compression_factor=None,
):
    cap = cv2.VideoCapture(video_path)
    if stop_frame is None:
        stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    _, frame = cap.read()
    frame_nrow, frame_ncol, _ = frame.shape
    using_crop, crop_settings = _parse_crop_settings(
        frame_nrow,
        frame_ncol,
        crop_interactive,
        crop_cmin,
        crop_cmax,
        crop_rmin,
        crop_rmax,
    )
    bg1 = cv2.createBackgroundSubtractorMOG2(history=MOG_history, detectShadows=False)
    num_frames = stop_frame - start_frame
    motion = np.zeros(num_frames - 1, dtype="uint32")
    for i in range(1, len(motion) + 1):
        frame_present, frame = cap.read()
        if frame_present:
            if using_crop:
                frame = _crop_frame(frame, *crop_settings)
            if compression_factor:
                frame = cv2.resize(
                    frame,
                    dsize=(
                        int(frame.shape[1] / compression_factor),
                        int(frame.shape[0] / compression_factor),
                    ),
                )
            frame = bg1.apply(frame)
            motion[i - 1] = np.count_nonzero(frame)
            frame = frame
        else:
            i -= 2  # Reset x to last frame detected
            motion = motion[:i]
            break
    cap.release()
    if use_med_filter:
        motion = medfilt(motion, med_filter_size)
    return motion


def detect_freezes(
    motion, freeze_threshold=0.5, min_duration=0,
):
    """
    Denote each frame as containg freezing or not.

    Args:
        motion (array-like): The motion numpy array. This contains the number of pixels exceding the
                             motion threshold and is obtained by the mfreeze.detect_motion function.
        freeze_threshold (float): The threshold for Freezing. Lower thresholds yeild more frames being
                                  classified as freezes.
        min_duration (int): Defines the minimum time period for a freeze. Freezes below this threshold will not
                            be counted as freezes.
    Returns:
        A numpy array with one element per frame. 0s denote absense of freezing and 1 denote freezing
    """

    freeze = np.zeros(len(motion))

    p_freeze = np.less(motion, freeze_threshold).astype(np.uint8)
    if min_duration:
        r = _runs(p_freeze, 1)
        r = r[(np.diff(r) > min_duration).flatten()]
        if len(r) == 0:
            return freeze
        for start, stop in r:
            freeze[start:stop] = 1
    else:
        freeze = p_freeze
    return freeze
