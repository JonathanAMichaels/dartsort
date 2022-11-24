import argparse
import numpy as np
import shutil

from pathlib import Path


ap = argparse.ArgumentParser()

ap.add_argument("input_binary")
ap.add_argument("--no-bad-channels", action="store_true", default=False)
ap.add_argument("-o", "--output", default=None)
args = ap.parse_args()


binary = Path(args.input_binary)
folder = binary.parent
standardized_file = args.output or folder / f"destriped_{binary.name}"


if not standardized_file.exists():
    destripe_raw_binary(binary, standardized_file, reject_channels=not args.no_bad_channels)
