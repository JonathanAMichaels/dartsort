import numpy as np
import shutil

from pathlib import Path
from tqdm.auto import trange

from neurodsp import voltage, utils
import spikeglx


def destripe_raw_binary(raw_bin_file, standardized_file, reject_channels=True):
    """
    Destripes a raw binary file and saves it to a standardized flat binary file.
    :param raw_bin_file:
    :param standardized_file:
    :return:
    """
    # run destriping
    standardized_file = Path(standardized_file)
    sr = spikeglx.Reader(raw_bin_file)
    h = sr.geometry
    batch_size_secs = 1
    batch_intervals_secs = 50

    # computes the z-score of the data by sampling and destriping a few batches
    nbatches = int(np.floor((sr.rl - batch_size_secs) / batch_intervals_secs - 0.5))
    wrots = np.zeros((nbatches, sr.nc - sr.nsync, sr.nc - sr.nsync))
    for ibatch in trange(nbatches, desc="destripe batches"):
        ifirst = int(
            (ibatch + 0.5) * batch_intervals_secs * sr.fs
            + batch_intervals_secs
        )
        ilast = ifirst + int(batch_size_secs * sr.fs)
        sample = voltage.destripe(
            sr[ifirst:ilast, : -sr.nsync].T, fs=sr.fs, neuropixel_version=1
        )
        np.fill_diagonal(
            wrots[ibatch, :, :],
            1 / utils.rms(sample) * sr.sample2volts[: -sr.nsync],
        )
    wrot = np.median(wrots, axis=0)

    standardized_file.parent.mkdir(parents=True, exist_ok=True)
    voltage.decompress_destripe_cbin(
        sr.file_bin,
        h=h,
        wrot=wrot,
        output_file=standardized_file,
        dtype=np.float32,
        nc_out=sr.nc - sr.nsync,
        reject_channels=reject_channels,
    )

    # also copy the companion meta-data file
    shutil.copy(sr.file_meta_data, standardized_file.with_suffix(".meta"))
    return standardized_file