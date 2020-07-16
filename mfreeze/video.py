import holoviews as hv
import cv2
import numpy as np
import warnings
from pathlib import Path
from holoviews import streams
from .utils import _parse_crop_settings, _crop_frame


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
    print(frame.shape)
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


def save_freeze_video(
    source_video_path,
    outfile_path,
    freezes=None,
    start_frame=0,
    stop_frame=None,
    crop_interactive=None,
    crop_cmin=None,
    crop_cmax=None,
    crop_rmin=None,
    crop_rmax=None,
    _subset_freezes=False,
):
    """
    Save original video anotated with the estimated freeze status of the tracked object

    Args:
        source_video_path (str): Path to the original video file used for location tracking.
        outfile_path (str): The name to use when saving the video.
        freezes (arraylike): The estimated freeze status of the tracked object for each frame. This array
                             can be obtained from the mfreeze.freezelib.detect_freezes function.
        start_frame (int): The first frame to use.
        stop_frame (int): The final frame to use. Defaults to the last frame.
        crop_interactive (holoviews.streams.BoxEdit): Holoviews stream object to use if using the interactive cropping
                                                      functionality. This can be obtained from the mfreeze.video.crop
                                                      function.
        crop_cmin (int): Optional value for manual cropping. Specifies minimum column value for each frame.
        crop_cmax (int): Optional value for manual cropping. Specifies maximum column value for each frame.
        crop_rmin (int): Optional value for manual cropping. Specifies minimum row value for each frame.
        crop_rmax (int): Optional value for manual cropping. Specifies maximum row values for each frame.
        _subset_freezes (bool): If using a subset of frames, whether to subset out the freezes array
    """
    cap = cv2.VideoCapture(source_video_path)
    stop_frame = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if stop_frame is None else stop_frame
    )
    num_frames = stop_frame - start_frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    _, frame = cap.read()
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_nrow, frame_ncol = frame.shape
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
        rmin, rmax, cmin, cmax = crop_settings
        video_size = (int(cmax - cmin), int(rmax - rmin))
    else:
        video_size = (int(frame_ncol), int(frame_nrow))

    suffix = Path(outfile_path).suffix
    if suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"xvid")
    else:
        raise ValueError(f"Unknown file suffix for outfile_path: {suffix}")

    writer = cv2.VideoWriter(outfile_path, fourcc, fps, video_size, False)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_pos = (10, 30)
    font_size = 1
    font_linetype = 2
    font_color = 255
    if freezes is not None:
        text = np.where(freezes == 1, "Freeze", "No Freeze")
        if _subset_freezes:
            end_frame = stop_frame + 1
            freezes = freezes[start_frame:end_frame]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(num_frames - 1):
        has_frame, frame = cap.read()
        if has_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if using_crop:
                frame = _crop_frame(frame, *crop_settings)
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


def save_loco_video(
    source_video_path,
    outfile_path,
    rc=None,
    start_frame=0,
    stop_frame=None,
    crop_interactive=None,
    crop_cmin=None,
    crop_cmax=None,
    crop_rmin=None,
    crop_rmax=None,
    _subset_rc=False,
):
    """
    Save original video with the estimated position of the tracked object annotated for each frame.

    The estimated position is marked with a crosshair.

    Args:
        source_video_path (str): Path to the original video file used for location tracking.
        outfile_path (str): The name to use when saving the video.
        rc (arraylike): The estimated position of the tracked object. Should be a (n_frames, 2) numpy array
                        with rows corresponding to the row, column position of the mouse for a given frame. This can
                        be generated using the mfreeze.freezelib.track_location function.
        start_frame (int): The first frame to use.
        stop_frame (int): The final frame to use. Defaults to the last frame.
        crop (holoviews.streams.BoxEdit): Holoviews stream object to use if using the interactive cropping
                                          functionality. This can be obtained from the mfreeze.video.crop function.
        crop_cmin (int): Optional value for manual cropping. Specifies minimum column value for each frame.
        crop_cmax (int): Optional value for manual cropping. Specifies maximum column value for each frame.
        crop_rmin (int): Optional value for manual cropping. Specifies minimum row value for each frame.
        crop_rmax (int): Optional value for manual cropping. Specifies maximum row values for each frame.
        _subset_rc (bool): If saving a subset of frames, whether to also subset the rc array.
    """
    cap = cv2.VideoCapture(source_video_path)
    stop_frame = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if stop_frame is None else stop_frame
    )
    num_frames = stop_frame - start_frame
    fps = cap.get(cv2.CAP_PROP_FPS)
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
        rmin, rmax, cmin, cmax = crop_settings
        video_size = (int(cmax - cmin), int(rmax - rmin))
    else:
        video_size = (int(frame_ncol), int(frame_nrow))
    suffix = Path(outfile_path).suffix
    if suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"xvid")
    else:
        raise ValueError(f"Unknown file suffix for outfile_path: {suffix}")

    writer = cv2.VideoWriter(outfile_path, fourcc, fps, video_size, False)

    if rc is not None and _subset_rc:
        end_frame = stop_frame + 1
        rc = rc[start_frame:end_frame]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(num_frames - 1):
        has_frame, frame = cap.read()
        if has_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if using_crop:
                frame = _crop_frame(frame, *crop_settings)
            if rc is not None:
                pos = (int(rc[i, 1]), int(rc[i, 0]))
                cv2.drawMarker(frame, position=pos, color=255)
            writer.write(frame)
        else:
            warnings.warn(f"Error: Only wrote {i} of {num_frames}")
            break
    writer.release()
    cap.release()
