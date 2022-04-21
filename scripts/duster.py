import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from pathlib import Path
from joblib.externals import loky
from tqdm.auto import tqdm, trange
import pickle

from spike_psvae import (
    cluster,
    merge_split_cleaned,
    cluster_viz_index,
    denoise,
    cluster_utils,
    triage,
    cluster_viz,
)

ap = argparse.ArgumentParser()

ap.add_argument("raw_data_bin")
ap.add_argument("residual_data_bin")
ap.add_argument("sub_h5")
ap.add_argument("output_dir")
ap.add_argument("--inmem", action="store_true")

args = ap.parse_args()

# %%
plt.rc("figure", dpi=200)

# %%
offset_min = 30
min_cluster_size = 25
min_samples = 25

# %%
raw_data_bin = Path(args.raw_data_bin)
assert raw_data_bin.exists()
residual_data_bin = Path(args.residual_data_bin)
assert residual_data_bin.exists()
sub_h5 = Path(args.sub_h5)
sub_h5.exists()

# %%
# raw_data_bin = Path("/mnt/3TB/charlie/re_snips/CSH_ZAD_026_snip.ap.bin")
# assert raw_data_bin.exists()
# residual_data_bin = Path("/mnt/3TB/charlie/re_snip_res/CSH_ZAD_026_fc/residual_CSH_ZAD_026_snip.ap_t_0_None.bin")
# assert residual_data_bin.exists()
# sub_h5 = Path("/mnt/3TB/charlie/re_snip_res/CSH_ZAD_026_fc/subtraction_CSH_ZAD_026_snip.ap_t_0_None.h5")
# sub_h5.exists()

# %%
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)


# %% tags=[]
# load features
with h5py.File(sub_h5, "r") as h5:
    spike_index = h5["spike_index"][:]
    x, y, z, alpha, z_rel = h5["localizations"][:].T
    maxptps = h5["maxptps"][:]
    z_abs = h5["z_reg"][:]
    geom_array = h5["geom"][:]
    firstchans = h5["first_channels"][:]
    end_sample = h5["end_sample"][()]
    start_sample = h5["start_sample"][()]
    print(start_sample, end_sample)

    recording_length = (end_sample - start_sample) // 30000

    start_sample += offset_min * 60 * 30000
    end_sample += offset_min * 60 * 30000
    channel_index = h5["channel_index"][:]
    z_reg = h5["z_reg"][:]

num_spikes = spike_index.shape[0]
end_time = end_sample / 30000
start_time = start_sample / 30000


# %%
(
    tx,
    ty,
    tz,
    talpha,
    tmaxptps,
    _,
    ptp_keep,
    idx_keep,
) = triage.run_weighted_triage(x, y, z_reg, alpha, maxptps, threshold=85)
idx_keep_full = ptp_keep[idx_keep]

# %%
# this will cluster and relabel by depth
features = np.c_[tx, tz, np.log(tmaxptps) * 30]
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size, min_samples=min_samples
)
clusterer.fit(features)

# z order
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)

# remove dups and re z order
clusterer, duplicate_ids = cluster_utils.remove_duplicate_units(
    clusterer, spike_index[idx_keep_full, 0], tmaxptps
)
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)

# labels in full index space (not triaged)
labels = np.full(x.shape, -1)
labels[idx_keep_full] = clusterer.labels_

# %%
z_cutoff = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
for za, zb in zip(z_cutoff, z_cutoff[1:]):
    fig = cluster_viz_index.array_scatter(
        clusterer.labels_,
        geom_array,
        tx,
        tz,
        tmaxptps,
        zlim=(za, zb),
    )
    fig.savefig(output_dir / f"B_pre_split_full_scatter_{za}_{zb}", dpi=200)
    plt.close(fig)

# %%
denoiser = denoise.SingleChanDenoiser().load()
device = "cpu"
denoiser.to(device)

# %%
templates = merge_split_cleaned.get_templates(
    raw_data_bin,
    geom_array,
    clusterer.labels_.max() + 1,
    spike_index[idx_keep_full],
    clusterer.labels_,
)

(
    template_shifts,
    template_maxchans,
    shifted_triaged_spike_index,
) = merge_split_cleaned.align_spikes_by_templates(
    clusterer.labels_, templates, spike_index[idx_keep_full]
)

# %%
shifted_full_spike_index = spike_index.copy()
shifted_full_spike_index[idx_keep_full] = shifted_triaged_spike_index

# %%
# split
h5 = h5py.File(sub_h5, "r"):
sub_wf = h5["subtracted_waveforms"]
if args.inmem:
    sub_wf = sub_wf[:]
labels_split = merge_split_cleaned.split_clusters(
    residual_data_bin,
    ,
    firstchans,
    shifted_full_spike_index,
    template_maxchans,
    template_shifts,
    labels,
    x,
    z_reg,
    maxptps,
    geom_array,
    denoiser,
    device,
)

# %%
# re-order again
clusterer.labels_ = labels_split[idx_keep_full]
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
labels = np.full(x.shape, -1)
labels[idx_keep_full] = clusterer.labels_

# %%
z_cutoff = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
for za, zb in zip(z_cutoff, z_cutoff[1:]):
    fig = cluster_viz_index.array_scatter(
        clusterer.labels_,
        geom_array,
        tx,
        tz,
        tmaxptps,
        zlim=(za, zb),
    )
    fig.savefig(output_dir / f"C_after_split_full_scatter_{za}_{zb}", dpi=200)
    plt.close(fig)

# %%
# get templates
templates = merge_split_cleaned.get_templates(
    raw_data_bin,
    geom_array,
    clusterer.labels_.max() + 1,
    spike_index[idx_keep_full],
    clusterer.labels_,
)

(
    template_shifts,
    template_maxchans,
    shifted_triaged_spike_index,
) = merge_split_cleaned.align_spikes_by_templates(
    clusterer.labels_, templates, spike_index[idx_keep_full]
)
shifted_full_spike_index = spike_index.copy()
shifted_full_spike_index[idx_keep_full] = shifted_triaged_spike_index

# %%
# merge
labels_merged = merge_split_cleaned.get_merged(
    residual_data_bin,
    sub_wf,
    firstchans,
    geom_array,
    templates,
    template_shifts,
    len(templates),
    shifted_full_spike_index,
    labels,
    x,
    z_reg,
    denoiser,
    device,
    distance_threshold=1.0,
    threshold_diptest=0.5,
    rank_pca=8,
    nn_denoise=True,
)

# %%
# re-order again
clusterer.labels_ = labels_merged[idx_keep_full]
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
labels = np.full(x.shape, -1)
labels[idx_keep_full] = clusterer.labels_

# %%
# final templates
templates = merge_split_cleaned.get_templates(
    raw_data_bin,
    geom_array,
    clusterer.labels_.max() + 1,
    spike_index[idx_keep_full],
    clusterer.labels_,
)

(
    template_shifts,
    template_maxchans,
    shifted_triaged_spike_index,
) = merge_split_cleaned.align_spikes_by_templates(
    clusterer.labels_, templates, spike_index[idx_keep_full]
)
shifted_full_spike_index = spike_index.copy()
shifted_full_spike_index[idx_keep_full] = shifted_triaged_spike_index


# save
np.save(output_dir / "labels.npy", labels)
cluster_centers.to_hdf(output_dir / "cluster_centers.h5")
pickle.dump(clusterer, output_dir / "clusterer.pickle")
np.svae(output_dir / "aligned_spike_index.npy", shifted_full_spike_index)
np.save(output_dir / "templates.npy", templates)
np.save(output_dir / "template_shifts.npy", template_shifts)
np.save(output_dir / "template_maxchans.npy", template_maxchans)


# %%
z_cutoff = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
for za, zb in zip(z_cutoff, z_cutoff[1:]):
    fig = cluster_viz_index.array_scatter(
        clusterer.labels_,
        geom_array,
        tx,
        tz,
        tmaxptps,
        zlim=(za - 50, zb + 50),
    )
    fig.savefig(output_dir / f"AAA_final_full_scatter_{za}_{zb}", dpi=200)
    plt.close(fig)

# %%
triaged_log_ptp = tmaxptps.copy()
triaged_log_ptp[triaged_log_ptp >= 27.5] = 27.5
triaged_log_ptp = np.log(triaged_log_ptp + 1)
triaged_log_ptp[triaged_log_ptp <= 1.25] = 1.25
triaged_ptp_rescaled = (triaged_log_ptp - triaged_log_ptp.min()) / (
    triaged_log_ptp.max() - triaged_log_ptp.min()
)
color_arr = plt.cm.viridis(triaged_ptp_rescaled)
color_arr[:, 3] = triaged_ptp_rescaled

# ## Define colors
unique_colors = [
    "#e6194b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#800000",
    "#aaffc3",
    "#808000",
    "#000075",
    "#000000",
]

cluster_color_dict = {}
for cluster_id in np.unique(clusterer.labels_):
    cluster_color_dict[cluster_id] = unique_colors[
        cluster_id % len(unique_colors)
    ]
cluster_color_dict[-1] = "#808080"  # set outlier color to grey

# %%
cluster_centers.index

# %%
sudir = Path(output_dir / "singleunit")
sudir.mkdir(exist_ok=True)


# plot cluster summary


def job(cluster_id):
    if (sudir / f"unit_{cluster_id:03d}.png").exists():
        return
    with h5py.File(sub_h5, "r") as d:
        fig = cluster_viz.plot_single_unit_summary(
            cluster_id,
            clusterer.labels_,
            cluster_centers,
            geom_array,
            200,
            3,
            tx,
            tz,
            tmaxptps,
            firstchans[idx_keep_full],
            spike_index[idx_keep_full, 1],
            spike_index[idx_keep_full, 0],
            idx_keep_full,
            d["cleaned_waveforms"],
            d["subtracted_waveforms"],
            cluster_color_dict,
            color_arr,
            raw_data_bin,
            residual_data_bin,
        )
        fig.savefig(sudir / f"unit_{cluster_id:03d}.png")
        plt.close(fig)


i = 0
with loky.ProcessPoolExecutor(
    12,
) as p:
    units = np.setdiff1d(np.unique(clusterer.labels_), [-1])
    for res in tqdm(p.map(job, units), total=len(units)):
        pass
