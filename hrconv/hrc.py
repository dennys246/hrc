import scipy.linalg, hrtree, json, mne
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def estimate_hrfs(nirx_folder, nirx_identifier, events, hrfs_filename = "hrf_estimates.json", **kwargs):
    """
    This function is the primary call for estimating an HRF across a subject pool
    fNIRS data. To accomplish this, the function creates a hrfunc.montage and for
    each nirx file found using the nirx_folder and nirx_identifier estimates an 
    event wise HRF then generates a channel wise distribution after deconvolving all
    available subjects.
    """
    # Set data context
    context = {
            'type': 'global',
            'doi': 'temp',
            'study': None,
            'task': None,
            'conditions': None,
            'stimulus': None,
            'duration': 12,
            'protocol': None,
            'age_range': None,
            'demographics': None
        }
    context = {**context, **kwargs} # Add user input

    # Grab all available nirx files
    nirx_files = glob(f"{nirx_folder}/**/{nirx_identifier}")
    
    _montage = montage(_load_fnirs(nirx_files[0]), hrfs_filename, **kwargs)

    for nirx_filename in nirx_files: # For each nirx object
        nirx_obj = _load_fnirs(nirx_filename) # Load the nirx

        _montage.deconvolve_hrf(nirx_obj, events) # Estimate the HRF

    _montage.generate_distribution(context['duration']) # Generate HRF distribution

    _montage.save(hrfs_filename, context['doi'], **kwargs) # Save montage

    return _montage


def locate_hrfs(nirx_obj, hrfs_filename = 'hrfs.json', **kwargs):
    """
    Locate local HRF's for the nirx object and return a montage with found HRF's

    Arguments:
        nirx_obj (mne raw object) - NIRS file loaded through mne
    """

    # Build a montage
    _montage = montage(nirx_obj, hrfs_filename, **kwargs)
    
    # Set default montage
    context = {
        'type': 'global',
        'doi': 'temp',
        'study': None,
        'task': None,
        'conditions': None,
        'stimulus': None,
        'duration': 12,
        'protocol': None,
        'age_range': None,
        'demographics': None
    }
    context = {**context, **kwargs} # Add user input
    
    _montage.localize_hrf() # Call to the montage to localize
    return _montage

def deconvolve_nirs(nirx_obj, events, hrfs_filename = "hrfs.json", verbose = True, **kwargs):
    """
    Deconvlve a fNIRS scan using estimated HRF's localized to optodes location

    Arguments:
        nirx_obj (mne raw object) - fNIRS scan loaded through mne
        events (list) - event impulse sequence of 0's and 1's
    """
    # Initialize a montage for the
    _montage = montage(nirx_obj, hrfs_filename, **kwargs)
        
    nirx_obj.load_data()

    # Define hrf deconvolve function to pass nirx object
    def hrf_deconvolution(nirx):
        original_len = nirx.shape[0]

        nirx = nirx / np.max(np.abs(nirx))

        # Regularization parameter (small value to stabilize)
        lambda_ = 5e-2

        # Convolution matrix
        A = scipy.linalg.toeplitz(nirx, expected_activity[:len(nirx)]).T

        # Solve with regularization
        #deconvolved_signal = np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ nirx)
        deconvolved_signal, _, _, _ = np.linalg.lstsq(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ nirx, rcond=None)

        lost_signal = [0.0 for _ in range(original_len - deconvolved_signal.shape[0])]
        
        if verbose:
            print(f"{len(nirx)} --> {deconvolved_signal.shape} | {len(lost_signal)} samples lost")
        return np.concatenate((deconvolved_signal, lost_signal))

    # Apply deconvolution and return the nirx object
    for ch_name, hrf in _montage.channels.items():
        expected_activity = _montage.convolve_hrf(events, hrf.trace)
        if ch_name == 'global':
            continue
        nirx_obj.apply_function(hrf_deconvolution, picks = [ch_name])
    return nirx_obj

def resample_nirs(nirx_obj, events, std_seed, hrfs_filename = "hrfs.json", verbose = True, **kwargs):
    """
    Resample fNIRs data for deep learning using the HRF confidence interval
    to establish a new HRF to deconvolve the NIRS data with.

    Arguments:
        nirx_obj (mne raw object) - fNIRS scan loaded through mne
        events (list) - event impulse sequence of 0's and 1's
        std_seed (float) - deviation to impart on HRF
    """
    # Initialize a montage for the
    _montage = montage(nirx_obj, hrfs_filename, **kwargs)
        
    nirx_obj.load_data()
    if verbose: # Check shape if verbose
        data = nirx_obj.get_data()
        print(f"Original fNIRS Length: {data.shape}\nMax Value: {np.max(data)}\nMin Value: {np.min(data)}\nAny Nan: {np.isnan(data).any()}\nAny inf: {np.isinf(data).any()}")

    print(f"Original HRF length:{hrf.shape}\nMax Value{max(hrf)}\nMin Value: {min(hrf)}\nAny Nan: {np.isnan(hrf).any()}\nAny inf: {np.isinf(hrf).any()}")

    # Define hrf deconvolve function to pass nirx object
    def hrf_deconvolution(nirx):
        original_len = nirx.shape[0]

        nirx = nirx / np.max(np.abs(nirx))

        # Regularization parameter (small value to stabilize)
        lambda_ = 5e-2

        # Convolution matrix
        A = scipy.linalg.toeplitz(nirx, np.hstack([expected_activity, np.zeros(len(nirx) - len(expected_activity))])).T

        # Solve with regularization
        #deconvolved_signal = np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ nirx)
        deconvolved_signal, _, _, _ = np.linalg.lstsq(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ nirx, rcond=None)

        lost_signal = [0.0 for _ in range(original_len - deconvolved_signal.shape[0])]
        
        if verbose:
            print(f"{len(nirx)} --> {deconvolved_signal.shape} | {len(lost_signal)} samples lost")
        return np.concatenate((deconvolved_signal, lost_signal))

    # Apply deconvolution and return the nirx object
    for ch_name, hrf in _montage.channels.items():
        if ch_name == 'global': # If a global HRF
            continue # skip

        # Resample HRF with passed in standard deviation seed
        resampled_hrf = hrf.resample(std_seed)

        # Convolve events with resampled HRF to assess expected activity
        expected_activity = _montage.convolve_hrf(events, resampled_hrf.trace)
        
        # Apply deconvolution to channel
        nirx_obj.apply_function(hrf_deconvolution, picks = [ch_name])
    
    return nirx_obj


class montage:

    """
    Class functions:
        - localize_hrf() - Tries to find a previously derived HRFs localized to the same region
        - convolve_hrf() - Convolves an impulse function (series of 0's and 1's) with an HRF
        - deconvolve_hrf() - Deconvolves a fNIRS signal and impulse function to derive the underlying HRF
        - generate_distribution() - Calculates an average HRF and it's standard deviation across time
        - save() - Saves the current montage HRFs
        - load() - Loads a montage of HRFs
    
    Class attributes:
        - nirx_obj (mne raw object) - NIRX object loaded in via MNE python library
        - sfreq (float) - Sampling frequency of the fNIRS object
        - channels (list) - fNIRS montage channel names
        - subject_estimates (list) - List of subject event-wise HRF estimate
        - channel_estimates (list) - List of channel HRF distribution estimates (position 0 is mean and 1 is std)
    """

    def __init__(self, nirx_obj, hrfs_filename = None, **kwargs):
        # Set data context
        context = {
                'type': 'global',
                'doi': 'temp',
                'study': None,
                'task': None,
                'conditions': None,
                'stimulus': None,
                'duration': 12,
                'protocol': None,
                'age_range': None,
                'demographics': None
        }
        context = {**context, **kwargs} # Add user input

        # Initialize an empty tree
        self.tree = hrtree.Tree(**kwargs)

        self.channels = {} # Create variable for holding poiners to each channel
        self.subject_estimates = {} # Create hold variable for storing intermediary data

        self.sfreq = nirx_obj.info['sfreq']

        if hrfs_filename: # If hrfs json filename provided, load in
            self.load(hrfs_filename)
        else:
            # Add empty HRF nodes to the tree for each HRF
            for channel in nirx_obj.info['chs']:
                # Grab pertinent info from nirx header
                ch_name = channel['ch_name']
                location = channel['loc'][:3]

                # create an empty HRF object
                empty_hrf = hrtree.HRF(
                    context['doi'],
                    ch_name, 
                    context['duration'], 
                    self.sfreq, 
                    [], 
                    [], 
                    location
                )

                # Add in subject level estimate variables 
                self.subject_estimates[ch_name] = {'events': [], 'estimates': []}

                # Insert empty hrf into tree and attach pointer to channel
                self.channels[ch_name] = self.tree.insert(empty_hrf)
    
    def localize_hrfs(self, max_distance = 0.5):
        """
        Tries to find local HRFs to each of the fNIRS optodes using the tree structure
        functionality to quickly find nearby HRF's. If it can't it will default to a
        global HRF estimated.

        Arguments:
            max_distance (float) - maximum distance in milimeter's a previously estimated HRF can be attached to an optode
        """
        
        for ch_name in self.channels.keys(): # Iterate through channels apart of nirx data
            
            hrf = self.tree.search_dfs(self.channels[ch_name], max_distance) # Search in space for similar HRF

            if hrf: # If found
                self.channels[ch_name].trace = hrf.trace # Add mean and std to montage for channel
                self.channels[ch_name].trace_std = hrf.trace_std

            else: # If hrf not found locally
                LookupError(f"Local HRF with given context couldn't be found for channel {ch_name}, searching for global HRF")
                # Adjust channel location temporarily to global node nexus -0.5
                ch_location = [self.channels[ch_name].x, self.channels[ch_name].y, self.channels[ch_name].z] 
                self.channels[ch_name].x, self.channels[ch_name].y, self.channels[ch_name].z = -0.5, -0.5, -0.5
                
                # Search for global HRF with similar context
                hrf = self.tree.search_bfs(self.channels[ch_name], max_distance)
                if hrf: # If found
                    self.channels[ch_name].trace = hrf.trace # Add mean and std to montage for channel
                    self.channels[ch_name].trace_std = hrf.trace_std
                else: # If global HRF not found
                    LookupError(f"Global HRF with given context could not be found for {ch_name}")
                
                # Replace global location with original optode location
                self.channels[ch_name].x, self.channels[ch_name].y, self.channels[ch_name].z = ch_location[0], ch_location[1], ch_location[2]

    def convolve_hrf(self, events, hrf):
        """ Convolve an event impulse series with an hrf to create an expected signal. Both
        inputs must be the same length.

        Argument:
            events (list) - List of 0's and 1's indicating event occurances
            hrf (list) - HRF estimated trace
        """
        return np.convolve(events, hrf, mode='full')

    def deconvolve_hrf(self, nirx_obj, events, duration = 12):
        """
        Estimate an HRF subject wise given a nirx object and event impulse series

        Arguments:
            nirx_obj (mne raw object) - fNIRS scan file loaded in through mne
            events (list) - Event impulse series indicating event occurences during fNIRS scan
        """
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
        
        self.generate_distribution(duration)

    def generate_distribution(self, duration = 12):
        """
        Calculate average and standard deviation of HRF across subjects for each channel

        Arguments:
            duration (float) - Duration in seconds of the HRF to estimate
        """

        # Check if HRF subject wise estimates have been calculated
        
        hrf_means = []
        hrf_stds = []

        # Generate channel wise HRF estimates
        length = int(self.sfreq * duration)
        for ch_name, estimates in self.subject_estimates.items():
            event_estimates = []
            for sub_ind, events in enumerate(estimates['events']):
                for event in events:
                    event_estimates.append(estimates['estimates'][sub_ind][event:event+length])
            hrf_mean = np.mean(event_estimates, axis = 0)
            hrf_std = np.std(event_estimates, axis = 0)
            
            # Add subject hrf estimates to channel
            self.channels[ch_name].trace = hrf_mean
            self.channels[ch_name].trace_std = hrf_std

            # Append mean and std of hrf estimate
            hrf_means.append(hrf_mean)
            hrf_stds.append(hrf_std)

        # Calculate global HRF mean and standard deviation
        global_mean = np.mean(hrf_means, axis = 0)
        global_std = np.mean(hrf_stds, axis = 0)

        # Create a global HRF variable
        global_hrf = hrtree.HRF(
            doi = self.channels[ch_name].context['doi'],
            ch_name = "global",
            duration = self.channels[ch_name].context['duration'],
            sfreq = self.sfreq,
            trace = global_mean,
            trace_std = global_std,
            location = [-0.5, -0.5, -0.5]
        )

        #Insert global hrf into tree and attach pointer to channels dict
        self.channels['global'] = self.tree.insert(global_hrf)
        
        # Plot all of the channel HRF estimates
        for channel, hrf in self.channels.items():
            hrf_mean = hrf.trace
            hrf_std = hrf.trace_std
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

        # Estimate global HRF?

    def save(self, json_filename, **kwargs):
        """ Save the HRF as a json using the json_filename"""

        self.context = {**self.context, **kwargs}
        doi = self.context['doi']

        # Format json
        json_contents = {}
        for ch_name in self.channels.keys():
            channel_identifier = f"{'-'.join(ch_name.split(' '))}-{doi}"
            json_contents[channel_identifier] = {
                'hrf_mean': self.channels[ch_name].trace.tolist(),
                'hrf_std': self.channels[ch_name].trace_std.tolist(),
                'location': [self.channels[ch_name].x, self.channels[ch_name].y, self.channels[ch_name].z],
                'sfreq': self.sfreq,
                'context': self.context
        }

        # Save to a JSON file
        with open(json_filename, "w") as file:
            json.dump(json_contents, file, indent=4)  # indent is optional, just makes it pretty
        return
    
    def load(self, json_filename):
        """ Load montage with the given json filename """
        # Read in json
        with open(json_filename, 'r') as file:
            json_contents = json.load(file)

        # Update montage with saved info
        for key, channel in json_contents.items():
            key_split = key.split('-')
            doi = key_split.pop()
            ch_name = ' '.join(key_split)

            # create an empty HRF object
            empty_hrf = hrtree.HRF(
                doi,
                ch_name, 
                channel['context']['duration'], 
                self.sfreq, 
                np.array(channel['hrf_mean']), 
                np.array(channel['hrf_std']), 
                channel['location']
            )

            # Insert empty hrf into tree and attach pointer to channel
            self.channels[ch_name] = self.tree.insert(empty_hrf)
        return
    
def _load_fnirs(nirs_filename):
    """ Load the fNIRS file based on the format found """
    if nirs_filename[-6:] == ".snirf": # If snirf format
        try:
            nirs_obj = mne.io.read_raw_snirf(nirs_filename)
        except:
            return ValueError("SNIRF file passed in failed to load")
    if nirs_filename[-1:] == "/": # If folder passed in
        try:
            nirs_obj = mne.io.read_raw_nirx(nirs_filename)
        except:
            return ValueError("NIRS folder failed to load")
    return nirs_obj
    
