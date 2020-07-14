import cv2
import numpy as np
from .utils import _parse_crop_settings, _crop_frame
import warnings
from scipy.ndimage.measurements import center_of_mass
from pathlib import Path


def reference_create(
    video_path,
    num_frames=200,
    start_frame=0,
    stop_frame=None,
    crop_interactive=None,
    crop_cmin=None,
    crop_cmax=None,
    crop_rmin=None,
    crop_rmax=None,
):
    """
    Create a reference image by averaging over many frames
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    stop_frame = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if stop_frame is None else stop_frame
    )

    frame_idx = np.random.randint(low=start_frame, high=stop_frame - 1, size=num_frames)
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
        frame = _crop_frame(frame, *crop_settings)
    frame_nrow, frame_ncol = frame.shape
    frame_mat = np.empty(shape=(num_frames, frame_nrow, frame_ncol))
    for i, ind in enumerate(frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
        frame_present, frame = cap.read()
        if frame_present:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if using_crop:
                frame = _crop_frame(frame, *crop_settings)
            frame_mat[i, :, :] = frame
        else:
            frame_mat = frame_mat[:i, :, :]
    ref_frame = np.median(frame_mat, axis=0)
    return ref_frame.astype("uint8")


def locate(
    video_path,
    ref_frame,
    thresh=99,
    start_frame=0,
    stop_frame=None,
    crop_interactive=None,
    crop_cmin=None,
    crop_cmax=None,
    crop_rmin=None,
    crop_rmax=None,
    method="abs",
):
    """
    Get the location of the object of interest
    """
    # TODO create reference within the function

    cap = cv2.VideoCapture(video_path)
    stop_frame = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if stop_frame is None else stop_frame
    )
    num_frames = stop_frame - start_frame
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

    x_y = np.empty((num_frames, 2)).astype("uint16")

    for i in range(num_frames - 1):
        frame_present, frame_new = cap.read()
        if frame_present:
            frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
            if using_crop:
                frame_new = _crop_frame(frame_new, *crop_settings)
            if method == "abs":
                diff = cv2.absdiff(ref_frame, frame_new)
            diff[diff < np.percentile(diff, thresh)] = 0
            x_y[i, :] = center_of_mass(diff)
    return x_y


def save_loco_video(
    source_video_path,
    outfile_path,
    xy=None,
    start_frame=0,
    stop_frame=None,
    video_size=(640, 480),
    crop_interactive=None,
    crop_cmin=None,
    crop_cmax=None,
    crop_rmin=None,
    crop_rmax=None,
):
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
    suffix = Path(outfile_path).suffix
    if suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"xvid")
    else:
        raise ValueError(f"Unknown file suffix for outfile_path: {suffix}")

    writer = cv2.VideoWriter(outfile_path, fourcc, fps, video_size, False)

    if xy is not None:
        end_frame = stop_frame + 1
        xy = xy[start_frame:end_frame]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(num_frames - 1):
        has_frame, frame = cap.read()
        if has_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if using_crop:
                frame = _crop_frame(frame, *crop_settings)
            if xy is not None:
                pos = (int(xy[i, 1]), int(xy[i, 0]))
                cv2.drawMarker(frame, position=pos, color=255)
            writer.write(frame)
        else:
            warnings.warn(f"Error: Only wrote {i} of {num_frames}")
            break
    writer.release()
    cap.release()
