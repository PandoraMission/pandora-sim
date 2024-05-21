import functools
from typing import Callable

from astropy.time import Time

PARAMETERS_DOCSTRINGS = {
    "SC_Resets1": (
        int,
        "Number of reset frames at the start of the first integration of exposure",
    ),
    "SC_Resets2": (
        int,
        "Number of resent frames at the start of 1 through n integrations of exposure",
    ),
    "SC_DropFrames1": (
        int,
        "Number of dropped frames after reset of any integration of exposure",
    ),
    "SC_DropFrames2": (
        int,
        "Number of dropped frames in every group of integrations of exposure except the last group",
    ),
    "SC_DropFrames3": (
        int,
        "Number of dropped frames in the last group of each integration of exposure",
    ),
    "SC_ReadFrames": (
        int,
        "Number of frames read during each group of integration of exposure",
    ),
    "SC_Groups": (int, "Number of groups per integration of exposure"),
    "SC_Integrations": (int, "Number of integrations per exposure"),
    "ra": (float, "Right Ascension of the pointing"),
    "dec": (float, "Declination of the pointing"),
    "theta": (float, "Roll angle of the pointing"),
    "ROI_size": (tuple, "Size in pixels of each region of interest"),
    "nROIs": (int, "Number of regions of interest"),
    "nreads": (
        int,
        "Number of reads to co-add together to make each frame. Default for Pandora VISDA is 50.",
    ),
    "nframes": (int, "Number of frames to return. Default is 100 frames."),
    "start_time": (
        Time,
        "Time to start the observation. This is only used if returning a fits file, or supplying a flux function.",
    ),
    "target_flux_function": (
        (Callable, None),
        "Function governing the generation of the time series from the subarray. Default is None.",
    ),
    "noise": (
        bool,
        "Flag determining whether noise is simulated and included. Default is True.",
    ),
    "jitter": (
        bool,
        "Flag determining whether jitter is simulated and included. Default is True.",
    ),
    "output_type": (
        str,
        'String flag to determine the result output. Valid strings are "fits" or "array"',
    ),
    "bin_frames": (
        int,
        "If `nreads` is high, many reads must be calculated per frame stored. To reduce this, set `bin_frames`. If `bin_frames=10`, only `nreads/10` reads will be calculated, each with an exposure time of `self.detector.integration_time * bin_frames`.",
    ),
}


# Decorator to add common parameters to docstring
def add_docstring(*param_names):
    def decorator(func):
        param_docstring = ""
        if func.__doc__:
            # Determine the current indentation level
            lines = func.__doc__.splitlines()
            if len(lines[0]) == 0:
                indent = len(lines[1]) - len(lines[1].lstrip())
            else:
                indent = len(lines[0]) - len(lines[0].lstrip())
        else:
            indent = 0
        indent_str = " " * indent
        for name in param_names:
            if name in PARAMETERS_DOCSTRINGS:
                dtype, desc = PARAMETERS_DOCSTRINGS[name]
                if isinstance(dtype, tuple):
                    dtype_str = " or ".join(
                        [t._name if hasattr(t, "_name") else t.__name__][0]
                        for t in dtype
                        if t is not None
                    )
                    dtype_str += " or None" if None in dtype else ""
                else:
                    dtype_str = dtype.__name__
                param_docstring += (
                    f"{indent_str}{name} : {dtype_str}\n{indent_str}    {desc}\n"
                )
        existing_docstring = func.__doc__ or ""
        if "Parameters" in existing_docstring:
            func.__doc__ = (
                existing_docstring.split("---\n")[0]
                + "---\n"
                + param_docstring
                + "---\n".join(existing_docstring.split("---\n")[1:])
            )
        else:
            func.__doc__ = (
                existing_docstring
                + f"\n{indent_str}Parameters:\n{indent_str}-----------\n"
                + param_docstring
            )
        return func

    return decorator


# Decorator to inherit docstring from base class
def inherit_docstring(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    if func.__doc__ is None:
        for base in func.__qualname__.split(".")[0].__bases__:
            base_func = getattr(base, func.__name__, None)
            if base_func and base_func.__doc__:
                func.__doc__ = base_func.__doc__
                break
    return wrapper
