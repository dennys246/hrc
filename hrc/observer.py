import mne, os
import numpy as np
from scipy.signal import welch
from mne_nirs.preprocessing import peak_power
from mne_nirs.visualisation import plot_timechannel_quality_metric
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis


class lens:
    # This object is primarily used to calculate summary statistics of the passed in and
    # preprocessed NIRX objects. This can be used to compare your output to published results
    def __init__(self, working_directory = None):
        
        # Set working directory is not passed in
        self.working_directory = working_directory or os.getcwd()

        self.metrics = {
            'Preprocessed': {
                'kurtosis': {},
                'skewness': {},
                'snr': {}
                },
            'Convolved': {
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
            response = meter(subject_id, preproc_nirx, 'Preprocessed')
            response = meter(subject_id, convolved_nirx, 'Convolved')


    def compare_subjects(self):
        channel_kurtosis = {state: {channel: 0 for channel in self.channels} for state in ['Preprocessed', 'Convolved']}
        channel_skewness = {state: {channel: 0 for channel in self.channels} for state in ['Preprocessed', 'Convolved']}
        
        kurtosis = {
            'Preprocessed': [],
            'Convolved': []
        }
        skewness = {
            'Preprocessed': [],
            'Convolved': []
        }
        for state in ['Preprocessed', 'Convolved']:
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
            plt.figure(figsize=(10, 8))

            # Set the number of bars
            bar_width = 0.2
            x = np.arange(len(channels))  # The x locations for the groups

            # Create the bar plot
            plt.bar(x - bar_width/2, metric['Preprocessed'], width=bar_width, label=f'Preprocessed {metric_name}', color='b', align='center')
            plt.bar(x + bar_width/2, metric['Convolved'], width=bar_width, label=f'Convolved {metric_name}', color='g', align='center')

            # Adding labels and title
            plt.xlabel('Positions')
            plt.ylabel('Values')
            plt.title(f'Effects of Convolution on {metric_name.lower()}')
            plt.xticks(x, channels, rotation='vertical')  # Set the position names as x-tick labels
            plt.legend()

            # Show the plot
            plt.savefig(f'{self.working_directory}/plots/channel_wise_{metric_name.lower()}.jpeg')
            plt.close()

        print(f"SCI: {self.metrics['SCI'].shape}")

        plt.hist(self.metrics['SCI'])
        plt.title(f'Scalp Coupling Index')
        plt.savefig(f'{self.working_directory}/plots/subject_wise_sci.jpeg')
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

        plt.savefig(f'{self.working_directory}/plots/plotted_channel_data.jpeg')
        plt.close()

    def calc_pp(self, subject_id, scan, state):
        print(f"Calculating peakpower for {state} data...")
        preproc_nirx = scan.load_data()

        preproc_od = mne.preprocessing.nirs.optical_density(preproc_nirx)
        preproc_od, scores, times = peak_power(preproc_od, time_window=10)

        figure = plot_timechannel_quality_metric(preproc_od, scores, times, threshold=0.1)
        plt.savefig(f'{self.working_directory}/plots/{state}_powerpeak.jpeg')
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
        plt.savefig(f'{self.working_directory}/plots/{state}_sci.jpeg')
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
        plt.title(f'{state} Power Spectral Density for {subject_id}')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
        plt.xlim(0, 3)  # Limit to frequencies of interest
        plt.yscale('log')  # Log scale for better visualization of peaks
        plt.savefig(f'{self.working_directory}/plots/{state}_hr_presence.jpeg')
        plt.close()
    
    # Individual waveforms per channel

