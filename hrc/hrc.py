import statistics, os
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter

def convolve_hrf(nirx_obj, filter = None, hrf_duration = None, filter_type = 'normal', mean_window = 2, sigma = 5, scaling_factor = 0.1, plot = False):

    # Create hrf filter
    nirx_obj.load_data()

    if filter == None:
        hrf_build = hrf(nirx_obj.info['sfreq'], filter, hrf_duration, filter_type, mean_window, sigma, scaling_factor, plot)
        filter = hrf_build.filter
    
    # Convolve our NIRX signals with the hrf filter using a fast Fourier transform
    hrf_convolution = lambda nirx : signal.fftconvolve(nirx, filter, mode = 'same')
    return nirx_obj.apply_function(hrf_convolution)

class hrf:
    # This object is intended to generate a synthetic hemodynamic response function to be
    # convovled with a NIRS object. You can pass in a variety of optional parameters like mean window,
    # sigma and scaling factor to alter the way your filter is generated.
    def __init__(self, freq, filter = None, hrf_duration = None, filter_type = 'normal', mean_window = 2, sigma = 5, scaling_factor = 0.1, plot = False, working_directory = None):
        self.freq = freq
        self.filter_type = filter_type
        self.mean_window = mean_window
        self.sigma = sigma
        self.scaling_factor = scaling_factor

        # Get working directory
        self.working_directory = working_directory or os.getcwd()
        
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
            self.filter = self.filters[self.filter_type.lower()]['base-filter']

            # Calculate number of samples per hemodynamic response function
            # Number of seconds per hrf (seconds/hrf) mutliplied by samples per seconds
            self.hrf_duration = self.filters[self.filter_type.lower()]['duration']
            self.hrf_samples = round((self.hrf_duration) * self.freq, 2)
        else:
            if hrf_duration == None:
                print("User defined hrf filter passed in without hrf duration being specified, filter cannot be defined without hrf duration being specified. Please pass this information hrf_duration with your call to continue...")
                return
            else:
                self.filter = filter
                self.hrf_samples = round((hrf_duration) * self.freq, 2)
                

        if plot: # Plot the base filter
                plt.plot(self.filter)
                plt.title(f'{self.filter_type} hrf Interval Averages') 
                plt.savefig(f'{self.working_directory}/plots/synthetic_hrf_base.jpeg')
                plt.close()
        
        self.build(self.filter, hrf_duration, plot)

    def localize(self):
        # Optode-filter clustering - cluster filter's 
        # using optode locations as the cluster center
        # Concentrically move around the optode location
        # till a filter is found. What should be the limit?
        return

    def build(self, filter = None, hrf_duration = None, plot = False):
        if filter != None:
            if hrf_duration == None:
                print('Filter passed into hrf.build() function without passing in hrf duration, to build a custom filter please provide the duration in seconds of your expected hrf...')
            else:
                self.filter = filter
                self.hrf_duration = hrf_duration
                self.hrf_samples = round((self.hrf_duration) * self.freq, 2)

        # Define the processes for generating an hrf
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
                plt.title(f'{process_name}ed hrf')
                plt.savefig(f'{self.working_directory}/plots/synthetic_hrf_{process_name.lower()}ed.jpeg')
                plt.close()

        return self.filter

    def expand(self, filter):
        # Continue to expand the filter until it's bigger than size we need

        print('Expanding hrf filter...')
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
        print(f'Compressing hrf with mean filter (window size of {window})...')
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
        
