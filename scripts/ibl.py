from spike_psvae import subtract, ibme
import h5py


def destripe():


def localize(standardized_bin, out_dir=None, fs=30000)
    file_h5 = subtract.subraction(
        standardized_bin,
        out_dir,
        n_sec_pca=20,
        sampling_rate=fs,
        thresholds=[12, 10, 8, 6, 5, 4],
        denoise_detect=True,
        neighborhood_kind='firstchan',
        enforce_decrease_kind='radial',
        localization_kind='logbarrier',
        n_jobs=8
    )
    return file_h5



def register(file_h5, fs=30000, n_windows=10):
    """
    :param file_h5:
    :return:
    """
    with h5py.File(file_h5, "r+") as h5:
        del h5["z_reg"]
        del h5["dispmap"]

        samples = h5["spike_index"][:, 0] - h5["start_sample"][()]
        z_abs = h5["localizations"][:, 2]
        maxptps = h5["maxptps"]

        z_reg, dispmap = ibme.register_nonrigid(
            maxptps,
            z_abs,
            samples / fs,
            corr_threshold=0.6,
            disp=200 * n_windows,
            denoise_sigma=0.1,
            rigid_init=False,
            n_windows=n_windows,
            widthmul=1.0,
        )
        z_reg -= (z_reg - z_abs).mean()
        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("dispmap", data=dispmap)