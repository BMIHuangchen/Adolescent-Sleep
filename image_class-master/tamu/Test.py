import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Authors: Britta Westner <britta.wstnr@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
from mne.datasets import sample, fetch_fsaverage
from mne.beamformer import make_lcmv, apply_lcmv

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Read the raw data
raw = mne.io.read_raw_fif(raw_fname)
raw.info['bads'] = ['MEG 2443']  # bad MEG channel

# Set up the epoching
event_id = 1  # those are the trials with left-ear auditory stimuli
tmin, tmax = -0.2, 0.5
events = mne.find_events(raw)

# pick relevant channels
raw.pick(['meg', 'eog'])  # pick channels of interest

# Create epochs
proj = False  # already applied
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), preload=True, proj=proj,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))

# for speed purposes, cut to a window of interest
evoked = epochs.average().crop(0.05, 0.15)

# Visualize averaged sensor space data
evoked.plot_joint()

del raw  # save memory

data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.25,
                                  method='empirical')
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0,
                                   method='empirical')
data_cov.plot(epochs.info)
del epochs

# Read forward model

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-vol-7-fwd.fif'
forward = mne.read_forward_solution(fwd_fname)

filters = make_lcmv(evoked.info, forward, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='unit-noise-gain', rank=None)

# You can save the filter for later use with:
# filters.save('filters-lcmv.h5')

filters_vec = make_lcmv(evoked.info, forward, data_cov, reg=0.05,
                        noise_cov=noise_cov, pick_ori='vector',
                        weight_norm='unit-noise-gain', rank=None)
# save a bit of memory
src = forward['src']
del forward

stc = apply_lcmv(evoked, filters, max_ori_out='signed')
stc_vec = apply_lcmv(evoked, filters_vec, max_ori_out='signed')
del filters, filters_vec

lims = [0.3, 0.45, 0.6]
kwargs = dict(src=src, subject='sample', subjects_dir=subjects_dir,
              initial_time=0.087, verbose=True)

stc.plot(mode='stat_map', clim=dict(kind='value', pos_lims=lims), **kwargs)