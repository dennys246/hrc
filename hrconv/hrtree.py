import os, json, random, hrhash
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import deque
from nilearn.glm.first_level import spm_hrf

class HRTree:
    """
    This object is intended to generate a synthetic hemodynamic response function to be
    convovled with a NIRS object. You can pass in a variety of optional parameters like mean window,
    sigma and scaling factor to alter the way your hrf is generated.

    Class attributes:

    Class functions:
    
    """
    def __init__(self, hrf_jsonfile = "hrfs.json", **kwargs):
        self.root = None

        self.hrf_jsonfile = hrf_jsonfile
        self.hrf_context = hrhash.HashTable(hrf_jsonfile)

        self.context = {
            'type': 'default',
            'doi': None,
            'study': None,
            'task': None,
            'conditions': None,
            'stimulus': None,
            'duration': None,
            'protocol': None,
            'age_range': None,
            'demographics': None
        }
        self.context = {**self.context, **kwargs}

    def build(self, hrf_jsonfile = None):

        if hrf_jsonfile == None: # If no json filename provided
            hrf_jsonfile = self.hrf_jsonfile # Set as class default

        hrfs_json = json.load(hrf_jsonfile) # Load HRFs from json
        
        for hrf in hrfs_json:
            # Check if the hrf matches the context
            self.insert(hrf)
    

    def insert(self, hrf, depth=0, node=None):
        """ Insert a new node into the 3D tree based on spatial position. """
        
        if node is None:
            node = self.root

        if self.root is None:
            self.root = hrf
            return self.root

        axis = depth % 3  # Cycle through x, y, z axes

        if (axis == 0 and hrf.x < node.x) or (axis == 1 and hrf.y < node.y) or (axis == 2 and hrf.z < node.z):
            if node.left is None:
                node.left = hrf
            else:
                self.insert(hrf, depth + 1, node.left)
        else:
            if node.right is None:
                node.right = hrf
            else:
                self.insert(hrf, depth + 1, node.right)

    def compare_context(self, **kwargs):
        self.context = {**self.context, **kwargs}

    
    def search_dfs(self, hrf, depth=0, node = None, max_distance = 0.5, max_point = None, min_point = None):
        if node is None:
            node = self.root

        if node is None: # Establish base case
            return None
        
        # Find max/min x, y and z if not calculated
        if max_point == None:
            min_point = [hrf.x - max_distance, hrf.y - max_distance, hrf.z - max_distance]
            max_point = [hrf.x + max_distance, hrf.y + max_distance, hrf.z + max_distance]
            

        if min_point[0] > node.x and min_point[1] > node.y and min_point[2] > node.z:
            # Check if right node
            if max_point[0] < node.x and max_point[1] < node.y and max_point[2] < node.z:
                return node

        axis = depth % 3
        if (axis == 0 and min_point[0] < node.x) or (axis == 1 and min_point[1] < node.y) or (axis == 2 and min_point[2] < node.z):
            return self.search_dfs(hrf, depth + 1, node.left, max_distance,)
        else:
            return self.search_dfs(hrf, depth + 1, node.right, max_distance)

    def search_bfs(self, hrf):
        if self.root is None:
            return None

        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.x == hrf.x and node.y == hrf.y and node.z == hrf.z:
                return node

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return None
      
    def delete(self, hrf):
        self.root = self._delete_recursive(self.root, hrf, 0)

    def _delete_recursive(self, node, hrf, depth):
        if node is None:
            return None

        axis = depth % 3

        if node.x == hrf.x and node.y == hrf.y and node.z == hrf.z:
            if node.right:
                min_node = self._find_min(node.right, axis, depth + 1)
                node.x, node.y, node.z, node.hrf_data = min_node.x, min_node.y, min_node.z, min_node.hrf_data
                node.right = self._delete_recursive(node.right, min_node.x, min_node.y, min_node.z, depth + 1)
            elif node.left:
                min_node = self._find_min(node.left, axis, depth + 1)
                node.x, node.y, node.z, node.hrf_data = min_node.x, min_node.y, min_node.z, min_node.hrf_data
                node.right = self._delete_recursive(node.left, min_node.x, min_node.y, min_node.z, depth + 1)
                node.left = None
            else:
                return None  # No children case

        elif (axis == 0 and hrf.x < node.x) or (axis == 1 and hrf.y < node.y) or (axis == 2 and hrf.z < node.z):
            node.left = self._delete_recursive(node.left, hrf, depth + 1)
        else:
            node.right = self._delete_recursive(node.right, hrf, depth + 1)

        return node

    def _find_min(self, node, axis, depth):
        if node is None:
            return None

        if depth % 3 == axis:
            if node.left is None:
                return node
            return self._find_min(node.left, axis, depth + 1)

        left_min = self._find_min(node.left, axis, depth + 1)
        right_min = self._find_min(node.right, axis, depth + 1)

        return min([node, left_min, right_min], key=lambda n: getattr(n, ["x", "y", "z"][axis]) if n else float('inf'))

class HRF:
    def __init__(self, trace, trace_std, duration, sfreq, location = None, plot = False, **kwargs):
        """
        Object for storing all information apart of an estimated HRF from an fNIRS optode

        Class functions:
            self.build() - Build the HRF to fit a new sampling frequency and run through processing requested
            self.spline_interp() - Resizes the HRF to new sampling frequency using spline interpolation
            self.smooth() - Smooths the HRF trace using a gaussian filter
            self.resample() - Resampled the HRF using the estimated HRF and it's standard deviation 
            self.plot() - Plots the current HRF trace attached to the class

        Class attributes:
            trace (list of floats) - A trace of the HRF
            trace_std (list of floats) - The standard deviation of the HRF over time
            duration (float) - Duration of the HRF in seconds
            sfreq (float) - Sampling frequency of the fNIRS device that the HRF estimate was recorded from
            location (list of floats) - Location of the optode the HRF was estimated from the fNIRS device
            plot (bool) - Request for whether to plot the HRF throughout it's preprocessing
            **kwargs - Context attributes to be updated, only used by class or developers

        """
        # Set working directory and create plot 
        self.working_directory = os.getcwd()
        if os.path.exists(f"{self.working_directory}/plots/") == False and plot:
            os.mkdir(f"{self.working_directory}/plots/")

        # Attach passed into info to class 
        self.sfreq = sfreq
        self.length = int(round(self.sfreq * duration, 0))

        if trace == None:
            self.trace = spm_hrf(self.sfreq)
        else:
            self.trace = trace
        self.trace_std = trace_std

        if location: # Grab location
            self.x = location[0]
            self.y = location[1]
            self.z = location[2]
        else:
            # If no location pass in, set to a random number between 0 and 1 to prevent a long tail
            self.x = -1 + random.random() 
            self.y = -1 + random.random()
            self.z = -1 + random.random()

        # Set HRF default context
        self.context = {
            'type': 'default',
            'doi': None,
            'study': None,
            'task': None,
            'conditions': None,
            'stimulus': None,
            'duration': duration,
            'protocol': None,
            'age_range': None,
            'demographics': None
        }
        unexpected = set(kwargs) - set(self.context)
        if unexpected:
            raise ValueError(f"Unexpected contexts cannot be added: {unexpected}\n\nMake sure the contexts your searching for are within the available contexts: {','.join(self.context.keys())}")
        self.context.update({key: value for key, value in kwargs.items() if key in self.context})

        self.left = None
        self.right = None

        self.hrf_processes = [self.spline_interp]
        self.process_names = ['spline_interpolate']
        self.process_options = []


    def build(self, new_sfreq, plot = False, show = False):
        # Define the processes for generating an hrf
        self.target_length = new_sfreq * float(self.context['duration'])
        for process, process_name, process_option in zip(self.hrf_processes, self.process_names, self.process_options):
            
            if process_option == None:
                self.trace = process(self.trace)
            else:
                self.trace = process(self.trace, process_option)
            
            if plot: # Plot the processing step results
                title = f"HRF - {process_name}"
                filename = f"plots/{'-'.join(process_name.split(' ')).lower()}_{self.type}_hrf_results.png"
                self.plot(title, filename, show)


    def spline_interp(self):
        """
        
        """
        # Original list
        hrf_indices = np.linspace(0, len(self.trace) - 1, len(self.trace))

        # Create a spline interpolation function
        spline = interp1d(hrf_indices, self.trace, kind='cubic')
        new_indices = np.linspace(0, len(self.trace) - 1, int(self.target_length))

        # Compressed list
        return spline(new_indices)

    def smooth(self, a):
        """
        Function that uses a gaussian filter to smooth the HRF trace.

        Function attributes:
            a (float) - Sigma value used in gaussian filter to dictate how much the HRF is smoothed
        """
        print(f'Smoothing HRF trace with Gaussian filter (sigma = {a})...')
        self.trace = self.gaussian_filter1d(self.trace, a)

        
    def normalize(self):
        """
        Function to normalize the trace between 0 and 1, useful for machine learning
        """
        self.trace = (self.trace - np.min(self.trace)) / (np.max(self.trace) - np.min(self.trace))

    def scale(self):
        """
        Function to scale around 1 using L2 normalization
        """
        self.trace /= np.linalg.norm(self.trace)
    
    def resample(self, std_seed = 0.0):
        """
        This resample function is an experimental resampling method for fNIRS (and potentially fMRI)
        for generating a new sample for machine learning and artificial intelligence training. The 
        general idea is to shift the HRF trace slightly within a confidence interval before deconvolving
        to generate multiple resampled fNIRS samples.

        Function attributes:
            std_seed (float) - Standard deviation seed between -3 and 3 to resample from the HRF trace deviation
        """
        if self.trace_deviation == None:
            raise ValueError(f"HRF does not have a trace deviation attached to it")
        # Resample trace
        return [mean + (std_seed * std) for mean, std in zip(self.trace, self.trace_std)]


    def plot(self, title = None, filepath = None, show = True):
        """
        # Function to plot the current HRF

        Function attributes:

        """

        plt.plot(self.trace) # Plot trace

        # Format plot
        plt.title(title) 
        
        if filepath: # If filepath provided
            plt.savefig(f'{self.working_directory}/plots/synthetic_hrf_base.jpeg') # Save

        elif show: # If plot requested to be shown
            plt.show() # Show plot

        plt.close() # Close all plots
        