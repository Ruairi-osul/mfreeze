import cv2
from pathlib import Path
import holoviews as hv
import numpy as np
from holoviews import streams
import warnings
from scipy.signal import medfilt
from .utils import _crop_frame, _runs, _parse_crop_settings


def interactive_crop(
    video_path, frame=0,
):
    """
    Loads and displays a frame for a video to be used for cropping.
    Cropping automatically updated using holoviews stream object.

    Args:
        video_path (str): Path to the video
        frame (int): The index of the frame to be used for cropping
    Returns:
        image, stream
    """

    hv.notebook_extension("bokeh")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()

    image = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
    image.opts(
        width=frame.shape[1],
        height=frame.shape[0],
        invert_yaxis=True,
        cmap="gray",
        colorbar=True,
        toolbar="below",
        title="First Frame.  Crop if Desired",
    )

    box = hv.Polygons([])
    box.opts(alpha=0.5)
    box_stream = streams.BoxEdit(source=box, num_objects=1)
    return (image * box), box_stream


def detect_motion(
    video_path,
    start_frame=0,
    stop_frame=None,
    crop_interactive=None,
    crop_cmin=None,
    crop_cmax=None,
    crop_rmin=None,
    crop_rmax=None,
    use_med_filter=True,
    med_filter_size=3,
):
    """
    Returns the number of pixels exeding the motion threshold in each frame.

    Args:
        video_path (str): Path to a the video file
        motion_threshold (float): The threshold value used for. Units are change in greyscale pixel
                          value between frames.
        start_frame (int): Frame index to use as intial frame.
        stop_frame (int): Frame to use as
        crop (holoviews.streams.BoxEdit): Holoviews stream object to use if cropping image.
                                          This can be obtained from mfreeze.crop
        gaussian_kernel (tuple): Kernel to use for gaussian smoothing of frames.
        gaussian_sigma (float): Smoothing parameter to use for gaussian smoothing of frames.
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if using_crop:
                frame = _crop_frame(frame, *crop_settings)
            motion[i - 1] = np.sum(cv2.absdiff(frame_old, frame))
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


def detect_freezes(
    motion, freeze_threshold, min_duration=0,
):
    """
    Denote each frame as containg freezing or not.

    Args:
        motion (array-like): The motion numpy array. This contains the number of pixels exceding the
                             motion threshold and is obtained by the mfreeze.detect_motion function.
        freeze_threshold (int): The threshold for Freezing. Units are the maximum number of pixels above the
                                motion threshold a frame may have in order to be denoted a freeze.
        min_duration (int): Defines the minimum time period for a freeze. Freezes below this threshold will not
                            be counted as freezes.
        med_filter (bool): Whether to apply a median filter to the motion array before detecting freezes.
        filter_size (int): If using the median filter, selects the kernel size.
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


def save_video(
    source_video_path,
    outfile_path,
    freezes=None,
    start_frame=0,
    stop_frame=None,
    video_size=(640, 480),
):
    cap = cv2.VideoCapture(source_video_path)
    if stop_frame is None:
        stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = stop_frame - start_frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    _, frame = cap.read()
    suffix = Path(outfile_path).suffix
    if suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"xvid")
    else:
        raise ValueError(f"Unknown file suffix for outfile_path: {suffix}")
    writer = cv2.VideoWriter(outfile_path, fourcc, fps, video_size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_pos = (10, 30)
    font_size = 1
    font_linetype = 2
    font_color = 255
    if freezes is not None:
        end_frame = stop_frame + 1
        freezes = freezes[start_frame:end_frame]
        text = np.where(freezes == 1, "Freeze", "No Freeze")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(num_frames - 1):
        has_frame, frame = cap.read()
        if has_frame:
            if freezes is not None:
                frame = cv2.putText(
                    frame, text[i], font_pos, font, font_size, font_color, font_linetype
                )
            writer.write(frame)
        else:
            warnings.warn(f"Error: Only wrote {i} of {num_frames}")
            break
    writer.release()
    cap.release()
