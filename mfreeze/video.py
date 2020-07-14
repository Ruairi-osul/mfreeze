import holoviews as hv
import cv2
import numpy as np
import warnings
from pathlib import Path
from holoviews import streams


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
