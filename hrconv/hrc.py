import statistics, os
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def convolve_hrf(nirx_obj, filter = None, hrf_duration = None, filter_type = 'normal', mean_window = 2, sigma = 5, scaling_factor = 0.1, plot = False):
    nirx_obj.load_data()

    # Create hrf filter
    if filter == None: # Create a hrf filter if none were passed in
        hrf_build = hrf(nirx_obj.info['sfreq'], filter, hrf_duration, filter_type, mean_window, sigma, scaling_factor, plot)
        filter = hrf_build.filter
    
    # Convolve our NIRX signals with the hrf filter using a fast Fourier transform
    hrf_convolution = lambda nirx : signal.fftconvolve(nirx, filter, mode = 'same')

    # Apply convolutional function and return the resulting nirx object
    return nirx_obj.apply_function(hrf_convolution)

def deconvolve_hrf(nirx_obj, filter = None, hrf_duration = None, filter_type = 'normal', mean_window = 2, sigma = 5, scaling_factor = 0.1, plot = False, verbose = False):

    # Load NIRX object data
    nirx_obj.load_data()
    if verbose: # Check shape if verbose
        data = nirx_obj.get_data()
        print(f"Original fNIRS Length:{data.shape}")
    
    if filter == None: # Create hrf filter if none were passed in
        hrf_build = hrf(nirx_obj.info['sfreq'], filter, hrf_duration, filter_type, mean_window, sigma, scaling_factor, plot)
        filter = hrf_build.filter
        filter /= np.sum(filter)

    # Define hrf deconvolve function to pass nirx object
    def hrf_deconvolution(nirx):
        original_len = nirx.shape[0]

        deconvolved_signal, remainder = signal.deconvolve(nirx, filter)
        lost_signal = [0.0 for point in range(original_len - deconvolved_signal.shape[0])]
        if verbose:
            print(f"{len(nirx)} --> {deconvolved_signal.shape} | {len(lost_signal)} samples lost")
        
        # Handle NaN values (replace NaNs with zeros or interpolate)
        deconvolved_signal = np.nan_to_num(deconvolved_signal, nan=0.0)s

        # Apply Ridge regression (L2 regularization)
        X = np.expand_dims(np.arange(len(deconvolved_signal)), axis=1)
        y = deconvolved_signal
        ridge = Ridge(alpha=1.0)  # Adjust alpha to control regularization strength
        ridge.fit(X, y)
        regularized_signal = ridge.predict(X)

        return np.concatenate((regularized_signal, lost_signal))


    # Apply deconvolution and return the nirx object
    return nirx_obj.apply_function(hrf_deconvolution)


class hrf:
    # This object is intended to generate a synthetic hemodynamic response function to be
    # convovled with a NIRS object. You can pass in a variety of optional parameters like mean window,
    # sigma and scaling factor to alter the way your filter is generated.
    def __init__(self, freq, filter = None, hrf_duration = None, filter_type = 'normal', mean_window = 2, sigma = 5, scaling_factor = 1, plot = False, working_directory = None):
        self.freq = freq
        self.filter_type = filter_type
        self.mean_window = mean_window
        self.sigma = sigma
        self.scaling_factor = scaling_factor

        # Set working directory and create plot 
        self.working_directory = working_directory or os.getcwd()
        if os.path.exists(f"{self.working_directory}/plots/") == False and plot:
            os.mkdir(f"{self.working_directory}/plots/")
        
        if filter == None: # If a filter was not passed in
            self.filters = {'normal' : {
                                'base-filter': [-0.004, -0.02, -0.05, 0.6, 0.6, 0, -0.1, -0.105, -0.09, -0.04, -0.01, -0.005, -0.001, -0.0005, -0.00001, -0.00001, -0.0000001],
                                'duration': 30
                            },
                            'double-gamma' : {
                                'base-filter': [0.0, 8.984895865256286e-05, 0.002123531159055297, 0.011909966631128265, 0.03706805845688154, 0.08354981939596492, 0.15354912820708197, 0.24511895987749144, 0.3529654667697545, 0.4697754706726706, 0.5875882711130136, 0.6989264311025679, 0.7975691001523737, 0.8789641647532551, 0.9403372313395328, 0.9805796942560802, 0.999998727986137, 1.0, 0.9827568871737806, 0.9509027063732469, 0.9072674377099942, 0.8546686354204567, 0.7957577884067325, 0.7329179299152075, 0.6682052217180874, 0.6033259388717743, 0.5396402024597174, 0.47818449445524425, 0.4197060930117107, 0.3647038388401432, 0.31347091712512376, 0.26613651233771735, 0.22270421049126743, 0.1830858631275171, 0.14713028962103242, 0.11464669243190473, 0.08542301395951162, 0.05923969676910561, 0.03587944446100096, 0.01513364013895716, -0.0031939172487605956, -0.019285334798499334, -0.03330844410525588, -0.04541763158295952, -0.05575527901514212, -0.06445349541205905, -0.07163589904368567, -0.07741927839431614, -0.08191502101961703, -0.08523024916007464, -0.08746864051558506, -0.08873094240291249, -0.08911520855753266, -0.08871680127913469, -0.08762820872916355, -0.08593872924768607, -0.08373407278037948, -0.08109592499841216, -0.07810151341814249, -0.07482320759317862, -0.07132817791076505, -0.06767813017935267, -0.06392912640809774, -0.060131496193770506, -0.056329838084852354, -0.05256310624115623, -0.048864774639319895, -0.04526306893085271, -0.0417812547477812, -0.038437970658929316, -0.03524759398565212, -0.03222062816657236, -0.029364101199058516, -0.02668196577310885, -0.02417549295589122, -0.021843652601458048, -0.019683474983697054, -0.01769038942914509, -0.015858536920839696, -0.014181054727733868, -0.012650332069461857, -0.011258236645164947, -0.009996312536414824, -0.008855950542191605, -0.007828532426607108, -0.006905550868605274, -0.006078707109940056, -0.005339988416875595, -0.0046817275159569, -0.0040966461481364135, -0.0035778848209794207, -0.0031190207370067986, -0.0027140757476237843, -0.002357516035374361, -0.002044245069974316, -0.0017695912219542567, -0.0015292912568298895, -0.001319470776455767, -0.0011366225255807435, -0.0009775833427250595],
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
                print('Filter passed into hrf.build() function without passing in HRF duration, to build a custom filter please provide the duration in seconds of your HRF...')
            else:
                self.filter = filter
                self.hrf_duration = hrf_duration
                self.hrf_samples = round((self.hrf_duration) * self.freq, 2)

        # Define the processes for generating an hrf
        hrf_processes = [self.expand, self.compress, self.smooth, self.scale]
        process_names = ['Expand', 'Compress', 'Smooth', 'Scal']
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

    def mean_compress(self, filter, window = 2): 
        # Compress the filter using a windowed mean filtering approach
        print(f'Compressing hrf with mean filter (window size of {window})...')
        while len(filter) > self.hrf_samples:
            filter = [statistics.mean(filter[ind:ind+window]) for ind in range(len(filter) - window)]
        return filter
    
    def interp_compress(self, filter):
        # Original indices
        original_indices = np.linspace(0, len(filter) - 1, len(filter))

        # New indices
        new_indices = np.linspace(0, len(filter) - 1, self.hrf_samples)

        return np.interp(new_indices, original_indices, filter)

    def compress(self, filter, window):
        # Original list
        filter_indices = np.linspace(0, len(filter) - 1, len(filter))

        # Create a spline interpolation function
        spline = interp1d(filter_indices, filter, kind='cubic')
        new_indices = np.linspace(0, len(filter) - 1, int(self.hrf_samples))

        # Compressed list
        return spline(new_indices)

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
        
