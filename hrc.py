import mne, statistics
import numpy as np
from scipy.signal import welch
from mne_nirs.preprocessing import peak_power
from mne_nirs.visualisation import plot_timechannel_quality_metric
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.stats import skew, kurtosis

class convolver:
    # This object is intended to be the primary facilitator of the HRF convolution when passed in a nirx object
    def __init__(self):
        return

    def convolve_hrf(self, nirx_obj, filter = None, hrf_duration = None, filter_type = 'normal', mean_window = 2, sigma = 5, scaling_factor = 0.1, plot = False):
        # Create HRF filter
        nirx_obj.load_data()
        hrf = HRF(nirx_obj.info['sfreq'], filter, hrf_duration, filter_type, mean_window, sigma, scaling_factor, plot)
        self.hrf_filter = hrf.filter
        
        # Convolve our NIRX signals with the HRF filter using a fast Fourier transform
        hrf_convolution = lambda nirx : signal.fftconvolve(nirx, self.hrf_filter, mode = 'same')
        return nirx_obj.apply_function(hrf_convolution)

class HRF:
    # This object is intended to generate a synthetic hemodynamic response function to be
    # convovled with a NIRS object. You can pass in a variety of optional parameters like mean window,
    # sigma and scaling factor to alter the way your filter is generated.
    def __init__(self, freq, filter = None, hrf_duration = None, filter_type = 'normal', mean_window = 2, sigma = 5, scaling_factor = 0.1, plot = False):
        self.freq = freq
        self.filter_type = filter_type
        self.mean_window = mean_window
        self.sigma = sigma
        self.scaling_factor = scaling_factor
        
        if filter == None: # If a filter was not passed in
            self.filters = {'normal' : {
                                'base-filter': [-0.004, -0.02, -0.05, 0.6, 0.6, 0, -0.1, -0.105, -0.09, -0.04, -0.01, -0.005, -0.001, -0.0005, -0.00001, -0.00001, -0.0000001],
                                'duration': 30
                            },
                            'undershootless': {
                                'base-filter': [ -0.0004, -0.0008, 0.6, 0.6, 0, -0.1, -0.105, -0.09, -0.04, -0.01, -0.005, -0.001, -0.0005, -0.00001, -0.00001, -0.0000001],
                                'duration': 30
                            },
                            'term-infant': {
                                'base-filter': [ -0.0004, -0.0008, 0.05, 0.1, 0.1, 0, -0.1, -0.105, -0.09, -0.04, -0.01, -0.005, -0.001, -0.0005, -0.00001, -0.00001, -0.0000001],
                                'duration': 30
                            },
                            'preterm-infant': {
                                'base-filter': [0, 0.08, 0.09, 0.1, 0.1, 0.09, 0.08, -0.001, -0.0005, -0.00001, -0.00005, -0.00001, -0.000005, -0.0000001],
                                'duration': 12
                            }
                        }
            self.filter_type = self.filter_type
            self.filter = self.filters[self.filter_type.lower()]['base-filter']

            # Calculate number of samples per hemodynamic response function
            # Number of seconds per HRF (seconds/HRF) mutliplied by samples per seconds
            self.hrf_duration = self.filters[self.filter_type.lower()]['duration']
            self.hrf_samples = round((self.hrf_duration) * self.freq, 2)
        else:
            if hrf_duration == None:
                print("User defined HRF filter passed in without hrf duration being specified, filter cannot be defined without hrf duration being specified. Please pass this information hrf_duration with your call to continue.")
                return
            else:
                self.filter = filter
                self.hrf_samples = round((hrf_duration) * self.freq, 2)
                

        if plot: # Plot the base filter
                plt.plot(self.filter)
                plt.title(f'{self.filter_type} HRF Interval Averages') 
                plt.savefig('synthetic_hrf_base.jpeg')
                plt.close()


    def build(self, filter = None, hrf_duration = None, plot = False):
        if filter != None:
            if hrf_duration == None:
                print('Filter passed into HRF.build() function without passing in hrf duration, to build a custom filter please provide the duration in seconds of your expected HRF...')
            else:
                self.filter = filter
                self.hrf_duration = hrf_duration
                self.hrf_samples = round((self.hrf_duration) * self.freq, 2)

        # Define the processes for generating an HRF
        hrf_processes = [self.expand, self.compress, self.smooth, self.scale]
        process_names = ['Expand', 'Compress', 'Smooth', 'Scale']
        process_options = [None, self.mean_window, self.sigma, self.scaling_factor]
        for process, process_name, process_option in zip(hrf_processes, process_names, process_options):
            if process_option == None:
                self.filter = process(self.filter)
            else:
                self.filter = process(self.filter, process_option)
            
            if plot: # Plot the processing step results
                plt.plot(self.filter)
                plt.title(f'{process_name}ed HRF')
                plt.savefig(f'synthetic_hrf_{process_name.lower()}ed.jpeg')
                plt.close()

        return self.filter

    def expand(self, filter):
        # Continue to expand the filter until it's bigger than size we need

        print('Expanding HRF filter...')
        while len(filter) < self.hrf_samples:
            # Define a new empty filter to add in expanded filter into
            new_filter = [] 
            # Iterate through the current filter
            for ind, data in enumerate(filter): 
                # Append the current data point
                new_filter.append(data) 
                # As long as theirs a datapoint in front to interpolate between
                if ind + 1 < len(filter): 
                    # Interpolate a data point in between current datapoint and next
                    new_filter.append((data + filter[ind + 1])/2)
            filter = new_filter
        return filter

    def compress(self, filter, window = 2): 
        # Compress the filter using a windowed mean filtering approach
        print(f'Compressing HRF with mean filter (window size of {window})...')
        while len(filter) > self.hrf_samples:
            filter = [statistics.mean(filter[ind:ind+window]) for ind in range(len(filter) - window)]
        return filter

    def smooth(self, filter, a = 5):
        # Smooth the filter using a Gaussian blur
        print('Smoothing filter with Gaussian filter (sigma = {a})...')
        return gaussian_filter(filter, sigma=a)

   
    def scale(self, filter, scaling_factor = 0.1):
        # Scale the filter by convolving a scalar with the filter
        print(f'Scaling filter by {scaling_factor}...')
        filter = np.array(filter)
        scalar = np.array([scaling_factor])
        return np.convolve(filter, scalar, mode = 'same')
        
class observer:
    # This object is primarily used to calculate summary statistics of the passed in and
    # preprocessed NIRX objects. This can be used to compare your output to published results
    def __init__(self):
        self.metrics = {
            'preprocessed': {
                'kurtosis': {},
                'skewness': {},
                'snr': {}
                },
            'convolved': {
                'kurtosis': {},
                'skewness': {},
                'snr': {}                
                },
            'SCI': []  
            }


    def compare_subject(self, subject_id, raw_nirx, preproc_nirx, convolved_nirx):
        self.channels = preproc_nirx.ch_names

        self.plot_nirx(subject_id, preproc_nirx, convolved_nirx)

        self.metrics['SCI'] = np.concatenate((self.metrics['SCI'], self.calc_sci(subject_id, raw_nirx, 'raw')), axis = 0)
        self.calc_pp(subject_id, raw_nirx, 'raw')

        meters = [self.calc_skewness_and_kurtosis, self.calc_snr, self.calc_heart_rate_presence]
        for meter in meters:
            response = meter(subject_id, preproc_nirx, 'preprocessed')
            response = meter(subject_id, convolved_nirx, 'convolved')


    def compare_subjects(self):
        channel_kurtosis = {state: {channel: 0 for channel in self.channels} for state in ['preprocessed', 'convolved']}
        channel_skewness = {state: {channel: 0 for channel in self.channels} for state in ['preprocessed', 'convolved']}
        
        kurtosis = {
            'preprocessed': [],
            'convolved': []
        }
        skewness = {
            'preprocessed': [],
            'convolved': []
        }
        for state in ['preprocessed', 'convolved']:
            count = 0
            #Add all kurtosis across subjects per channel
            for subject_id, channels in self.metrics[state]['kurtosis'].items():
                for channel in channels:
                    channel_kurtosis[state][channel] += self.metrics[state]['kurtosis'][subject_id][channel]
                    count += 1

            # Add all skewness across subjects per channel
            for subject_id, channels in self.metrics[state]['skewness'].items():
                for channel in channels:
                    channel_skewness[state][channel] += self.metrics[state]['skewness'][subject_id][channel]
            
            # Average across subjects for each channel
            for channel in channels:
                skewness[state].append(channel_skewness[state][channel] / count)
                kurtosis[state].append(channel_kurtosis[state][channel] / count) 

        for metric, metric_name in zip([kurtosis, skewness], ['Kurtosis', 'Skewness']):    
            # Set the number of bars
            bar_width = 0.2
            x = np.arange(len(channels))  # The x locations for the groups

            # Create the bar plot
            plt.bar(x - bar_width/2, metric['preprocessed'], width=bar_width, label=f'Preprocessed {metric_name}', color='b', align='center')
            plt.bar(x + bar_width/2, metric['convolved'], width=bar_width, label=f'Convolved {metric_name}', color='g', align='center')

            # Adding labels and title
            plt.xlabel('Positions')
            plt.ylabel('Values')
            plt.title(f'Effects of Convolution on {metric_name.lower()}')
            plt.xticks(x, channels)  # Set the position names as x-tick labels
            plt.legend()

            # Show the plot
            plt.savefig(f'channel_wise_{metric_name.lower()}.jpeg')
            plt.close()

        print(f"SCI: {self.metrics['SCI'].shape}")

        plt.hist(self.metrics['SCI'])
        plt.title(f'Subject-Wise Scalp Coupling Index')
        plt.savefig(f'subject_wise_sci.jpeg')
        plt.close()

    def plot_nirx(self, subject_id, preproc_scan, convolved_scan, channel = 1):
        preproc_scan.load_data()
        convolved_scan.load_data()

        preproc_data = preproc_scan.get_data([channel])
        convovled_data = convolved_scan.get_data([channel])

        plt.figure(figsize=(10, 6)) 

        plt.plot(preproc_data[0, :500], color='blue', label='Preprocessed NIRS data')
        plt.plot(convovled_data[0, :500], color='orange', label='Convolutioned NIRS data')
        
        plt.xlabel('Samples')
        plt.ylabel('µmol/L')
        plt.title(f'fNIRS channel data')

        plt.legend(loc='best')

        plt.savefig(f'plotted_channel_data.jpeg')
        plt.close()

    def calc_pp(self, subject_id, scan, state):
        print(f"Calculating peakpower for {state} data...")
        preproc_nirx = scan.load_data()

        preproc_od = mne.preprocessing.nirs.optical_density(preproc_nirx)
        preproc_od, scores, times = peak_power(preproc_od, time_window=10)

        figure = plot_timechannel_quality_metric(preproc_od, scores, times, threshold=0.1)
        plt.savefig(f'{state}_powerpeak.jpeg')
        plt.close()
        return True

    def calc_sci(self, subject_id, scan, state):
        # Load the nirx object
        preproc_nirx = scan.load_data()

        preproc_od = mne.preprocessing.nirs.optical_density(preproc_nirx)
        preproc_sci = mne.preprocessing.nirs.scalp_coupling_index(preproc_od)

        figure, axis = plt.subplots(1, 1)

        axis.hist(preproc_sci)
        axis.set_title(f'{subject_id} {state} Scalp Coupling Index')
        plt.savefig(f'{state}_sci.jpeg')
        plt.close()
        return preproc_sci

    def calc_snr(self, subject_id, scan, state):
        # Load the nirx object
        raw = scan.load_data()

        # Filter the raw data to obtain the signal and noise components
        # Define the signal band (i.e., hemodynamic response function band)
        signal_band = (0.01, 0.2)
        # Define the noise band (outside of the hemodynamic response)
        noise_band = (0.2, 1.0) 

        # Extract the signal in the desired band
        preproc_signal = raw.copy().filter(signal_band[0], signal_band[1], fir_design='firwin')

        # Extract the noise in the out-of-band frequency range
        preproc_noise = raw.copy().filter(noise_band[0], noise_band[1], fir_design='firwin')

        # Calculate the Power Spectral Density (PSD) for signal and noise using compute_psd()
        psd_signal = preproc_signal.compute_psd(fmin=signal_band[0], fmax=signal_band[1])
        psd_noise = preproc_noise.compute_psd(fmin=noise_band[0], fmax=noise_band[1])

        # Extract the power for each component
        signal_power = psd_signal.get_data().mean(axis=-1)  # Average power across frequencies for signal
        noise_power = psd_noise.get_data().mean(axis=-1)    # Average power across frequencies for noise

        # Calculate SNR
        snr = signal_power / noise_power
        snr = sum(snr)/len(snr)
        print(f"{state} signal-to-noise ratio - {snr}")
        self.metrics[state]['snr'][subject_id] = snr
        return snr

    def calc_skewness_and_kurtosis(self, subject_id, scan, state):
        # Load your raw NIRX data (assuming `raw` is already loaded)
        raw = scan.load_data()

        # Extract the time series data for each channel
        data = raw.get_data()  # shape: (n_channels, n_times)

        # Compute skewness and kurtosis for each channel
        skewness = skew(data, axis=1)  # Calculate skewness along the time dimension
        kurtosis_vals = kurtosis(data, axis=1)  # Calculate kurtosis along the time dimension

        # Display the results for each channel
        channel_skewness = {}
        channel_kurtosis = {}
        for ch_name, skew_val, kurt_val in zip(raw.ch_names, skewness, kurtosis_vals):
            channel_skewness[ch_name] = skew_val
            channel_kurtosis[ch_name] = kurt_val
            print(f"{state} - Channel {ch_name}: Skewness = {skew_val:.3f}, Kurtosis = {kurt_val:.3f}")

            if subject_id not in self.metrics[state]['skewness'].keys():
                self.metrics[state]['skewness'][subject_id] = {}
                self.metrics[state]['kurtosis'][subject_id] = {}
            self.metrics[state]['skewness'][subject_id][ch_name] = skew_val
            self.metrics[state]['kurtosis'][subject_id][ch_name] = kurt_val

    def calc_heart_rate_presence(self, subject_id, scan, state):
        # Assuming `raw_haemo` is your preprocessed fNIRS object with hemoglobin concentration data

        # Step 1: Define heart rate frequency range
        heart_rate_low = 0.8  # Lower bound in Hz
        heart_rate_high = 2.0  # Upper bound in Hz

        # Step 2: Calculate Power Spectral Density (PSD) for each channel
        sfreq = scan.info['sfreq']  # Sampling frequency
        n_per_seg = int(4 * sfreq)  # Length of each segment for Welch's method

        psd_list = []
        freqs, psd_all_channels = [], []

        # Compute PSD for each channel
        for i, channel_data in enumerate(scan.get_data()):
            freqs, psd = welch(channel_data, sfreq, nperseg=n_per_seg)
            psd_all_channels.append(psd)

        # Step 3: Plot PSD for each channel with heart rate range highlighted
        plt.figure(figsize=(12, 8))

        for i, psd in enumerate(psd_all_channels):
            plt.plot(freqs, psd, label=f'Channel {i+1}')
        
        # Highlight the heart rate frequency range
        plt.axvspan(heart_rate_low, heart_rate_high, color='red', alpha=0.2, label='Heart Rate Range (0.8-2.0 Hz)')

        # Customize plot
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (PSD)')
        plt.title(f'Power Spectral Density for {subject_id}')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
        plt.xlim(0, 3)  # Limit to frequencies of interest
        plt.yscale('log')  # Log scale for better visualization of peaks
        plt.savefig(f'{state}_hr_presence.jpeg')
        plt.close()
    
    # Individual waveforms per channel

    # SCI across subjects

    # Write our intro/discussion and describe plots in powerpoint
    # --> User Neurophotonics as structure
