"""
Runs the subtraction pipeline for an IBL insertion.

Runs:
    1. Destriping of the raw binary file
    2. Spike subtraction
    3. Drift registration

Folder architecture for input data:
├── raw_ephys_data
│         ├── probe00
│         │         ├── _spikeglx_ephysData_g0_t0.imec0.ap.cbin
│         │         ├── _spikeglx_ephysData_g0_t0.imec0.ap.ch
│         │         ├── _spikeglx_ephysData_g0_t0.imec0.ap.meta
│         │         ├── _spikeglx_ephysData_g0_t0.imec0.sync.npy
│         │         ├── _spikeglx_sync.channels.probe00.npy
│         │         ├── _spikeglx_sync.polarities.probe00.npy
│         │         └── _spikeglx_sync.times.probe00.npy

Folder architecture of output data:
└── spike_sorters
    └── psvae
        └── probe00
            ├── destriped__spikeglx_ephysData_g0_t0.imec0.ap.bin
            └── destriped__spikeglx_ephysData_g0_t0.imec0.ap.meta


example run:
python ./ibl_run.py /mnt/s1/pykilosort_reruns/angelakilab/Subjects/NYU-37/2021-01-30/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin
"""
import argparse
from pathlib import Path

import spikeglx
from spike_psvae import subtract, ibme, preprocessing
from iblutil.util import get_logger
from one.alf.files import get_session_path

logger = get_logger('spike_psvae', 'INFO')
ap = argparse.ArgumentParser()
ap.add_argument("input_binary")
args = ap.parse_args()

cbin_file = Path(args.input_binary)
sr = spikeglx.Reader(cbin_file)
pname = cbin_file.parent.name
psvae_folder = get_session_path(cbin_file).joinpath('spike_sorters', 'psvae', pname)

standardized_file = psvae_folder.joinpath(f"destriped_{cbin_file.name}").with_suffix('.bin')

if not standardized_file.exists():
    preprocessing.destripe_raw_binary(binary, standardized_file, reject_channels=not args.no_bad_channels)
else:
    logger.info(f"Standardized file already exists: {standardized_file} - skipping destriping")


sub_h5 = subtract.subtraction(
        standardized_file,
        psvae_folder,
        n_sec_pca=20,
        sampling_rate=int(sr.fs),
        thresholds=[12, 10, 8, 6, 5, 4],
        denoise_detect=True,
        neighborhood_kind="firstchan",
        enforce_decrease_kind="radial",
        localization_kind="logbarrier",
        overwrite=True,
        n_jobs=8,
    )
