import scipy.linalg, hrf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def deconvolve_nirs(nirx_obj, hrf = None, verbose = True):

    # Load NIRX object data
    nirx_obj.load_data()
    if verbose: # Check shape if verbose
        data = nirx_obj.get_data()
        print(f"Original fNIRS Length: {data.shape}\nMax Value: {np.max(data)}\nMin Value: {np.min(data)}\nAny Nan: {np.isnan(data).any()}\nAny inf: {np.isinf(data).any()}")
    
    if hrf == None: # Create hrf if none was passed in
        hrf_montage = montage(nirx_obj)

    print(f"Original hrf length:{hrf.shape}\nMax Value{max(hrf)}\nMin Value: {min(hrf)}\nAny Nan: {np.isnan(hrf).any()}\nAny inf: {np.isinf(hrf).any()}")

    # Define hrf deconvolve function to pass nirx object
    def hrf_deconvolution(nirx):
        original_len = nirx.shape[0]

        nirx = nirx / np.max(np.abs(nirx))

        # Regularization parameter (small value to stabilize)
        lambda_ = 5e-2

        # Convolution matrix
        A = scipy.linalg.toeplitz(nirx, np.hstack([hrf, np.zeros(len(nirx) - len(hrf))])).T

        # Solve with regularization
        #deconvolved_signal = np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ nirx)
        deconvolved_signal, _, _, _ = np.linalg.lstsq(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ nirx, rcond=None)

        lost_signal = [0.0 for _ in range(original_len - deconvolved_signal.shape[0])]
        
        if verbose:
            print(f"{len(nirx)} --> {deconvolved_signal.shape} | {len(lost_signal)} samples lost")
        return np.concatenate((deconvolved_signal, lost_signal))

    # Apply deconvolution and return the nirx object
    for optode, info in hrf_montage.optodes.items():
        hrf = info['hrf'] # Grab the HRF to be used for the optode
        nirx_obj.apply_function(hrf_deconvolution, picks = [optode])
    
    return nirx_obj


def deconvolve_pool(nirx_folder, nirx_identifier):
    # Grab all available nirx files
    nirx_files = glob(f"{nirx_folder}/**/{nirx_identifier}")

    # For each nirx object

        # Load the nirx

        # Estimate the HRF

    # Generate HRF distribution

    # For each nirx object

        # For each channel

            # Deconvolve channel

        # Save as a new nirx file

    return

class montage:

    def __init__(self, nirx_obj):
        self.nirx_obj = nirx_obj
        self.sfreq = nirx_obj.info['sfreq']
        self.optodes = {optode['ch_name']: {"location": optode['loc'][:3], "hrf": None} for optode in nirx_obj.info['chs']}
        self.subject_estimates = {optode['ch_name']: {'estimates': [], 'events': []} for optode in nirx_obj.info['chs']}
    
    def convolve_hrf(self, events, hrf):
        return np.convolve(events, hrf, mode='full')[:events.shape[0]]

    def deconvolve_hrf(self, nirx_obj, events):
        events = np.array(events)

        nirx_obj.load_data()
        data = nirx_obj.get_data()

        transformed_events = scipy.fft.fft(events)   # Fourier Transform of Neural Activity
        for fnirs_signal, channel in zip(data[:], nirx_obj.info['chs']) : # For each channel
            transformed_signal = scipy.fft.fft(fnirs_signal)   # Fourier Transform of fMRI Signal

            # Avoid division by zero by adding a small regularization term
            epsilon = 1e-6
            freq_deconv = (transformed_signal * np.conj(transformed_events)) / (np.abs(transformed_events) ** 2 + epsilon)  # Deconvolution in Frequency Domain

            # Compute the Inverse Fourier Transform to get estimated HRF
            hrf_estimate = np.real(scipy.fft.ifft(freq_deconv))

            self.subject_estimates[channel['ch_name']]['estimates'].append(hrf_estimate)
            self.subject_estimates[channel['ch_name']]['events'].append(events)
        return 
    
    def localize_hrf(self):
        self.hrtree = hrf.HRTree()
        for optode, context in self.optodes.items():
            self.hrtree.search_bfs()

    def generate_distribution(self, duration = 12):
        self.channel_estimates = {channel: [[], []] for channel in self.nirx_obj.info['ch_names']}

        # Generate channel wise HRF estimates
        length = int(self.sfreq * duration)
        for channel, estimates in self.subject_estimates.items():
            event_estimates = []
            for sub_ind, events in enumerate(estimates['events']):
                for event in events:
                    event_estimates.append(estimates['estimates'][sub_ind][event:event+length])
            hrf_mean = np.mean(event_estimates, axis = 0)
            hrf_std = np.std(event_estimates, axis = 0)

            self.channel_estimates[channel][0] = hrf_mean
            self.channel_estimates[channel][1] = hrf_std
        
        for channel, estimates in self.channel_estimates.items():
            hrf_mean = estimates[0]
            hrf_std = estimates[1]
            time = np.arange(len(hrf_mean))

            plt.figure(figsize=(8, 4))
            plt.plot(time, hrf_mean, label='Mean HRF', color='blue')
            plt.fill_between(time, hrf_mean - hrf_std, hrf_mean + hrf_std, color='blue', alpha=0.3, label='±1 SD')

            plt.xlabel('Samples')
            plt.ylabel('HRF amplitude')
            plt.title(f'Estimated HRF for {channel} with Standard Deviation')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"/storage1/fs1/perlmansusan/Active/moochie/github/hrc/plots/{channel}_hrf_estimate.png")
