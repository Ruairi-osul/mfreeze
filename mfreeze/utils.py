import numpy as np


def _runs(arr, val):
    """
    Return a (n, 2) array where rows correspond to the start and stop
    indexes of runs of consecutive values in arr

    Args:
        arr (array-like): array in which to search for runs
        value (float): value for which runs will be searched
    """
    is_val = np.equal(arr, val).view(np.uint8)
    is_val = np.concatenate((np.array([0]), is_val, np.array([0])))
    absdiff = np.abs(np.diff(is_val))
    return np.where(absdiff == 1)[0].reshape(-1, 2)


def _crop_frame(frame, rmin, rmax, cmin, cmax):
    return frame[rmin:rmax, cmin:cmax]


def _unpack_hv_crop(crop_object):
    """
    """
    Xs = [crop_object.data["x0"][0], crop_object.data["x1"][0]]
    Ys = [crop_object.data["y0"][0], crop_object.data["y1"][0]]
    cmin, cmax = int(min(Xs)), int(max(Xs))
    rmin, rmax = int(min(Ys)), int(max(Ys))
    return rmin, rmax, cmin, cmax


def _parse_crop_settings(
    frame_nrow,
    frame_ncol,
    crop_interactive=None,
    cmin=None,
    cmax=None,
    rmin=None,
    rmax=None,
):
    has_hard_coded = (
        rmin is not None or rmax is not None or cmin is not None or cmax is not None
    )
    if crop_interactive and has_hard_coded:
        raise ValueError(
            "Cannot pass both interactive crop object and hard coded crop options"
        )
    if crop_interactive is not None:
        rmin, rmax, cmin, cmax = _unpack_hv_crop(crop_interactive)
        return True, (rmin, rmax, cmin, cmax)
    elif has_hard_coded:
        cmin = cmin if cmin is not None else 0
        cmax = cmax if cmax is not None else frame_ncol
        rmin = rmin if rmin is not None else 0
        rmax = rmax if rmax is not None else frame_nrow
        return True, (rmin, rmax, cmin, cmax)
    else:
        return False, (None, None, None, None)


def crop_set_same(cropped_processor, to_crop_processor):
    """
    Use same cropping locations between processors
    """
    to_crop_processor.cmin = cropped_processor.cmin
    to_crop_processor.cmax = cropped_processor.cmax
    to_crop_processor.rmin = cropped_processor.rmin
    to_crop_processor.rmax = cropped_processor.rmax
    return to_crop_processor
