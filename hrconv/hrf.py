import os, json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import deque
from nilearn.glm.first_level import spm_hrf

class tree:
    """
    This object is intended to generate a synthetic hemodynamic response function to be
    convovled with a NIRS object. You can pass in a variety of optional parameters like mean window,
    sigma and scaling factor to alter the way your hrf is generated.

    Class attributes:

    Class functions:
    
    """
    def __init__(self, hrf_json = "hrfs.json"):
        self.lexicon = {  # Create variable for storing context pointers
            'doi': {},
            'task': {},
            'conditions': {},
            'stimulus': {},
            'duration': {},
            'protocol': {},
            'age_range': {},
            'demographics': {}}
        
        hrfs_json = json.load(hrf_json) # Load HRFs from json
        
        for hrf in hrfs_json:
            self.insert(hrf)

        self.hash_stage =  0
        self.hash_order = [
            'doi',
            'task',
            'conditions',
            'stimulus',
            'duration',
            'protocol',
            'age_range',
            'demographics'
        ]

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

    def search_dfs(self, hrf, depth=0, node=None):
        if node is None:
            node = self.root

        if node is None:
            return None
        if node.x == hrf.x and node.y == hrf.y and node.z == hrf.z:
            return node

        axis = depth % 3
        if (axis == 0 and hrf.x < node.x) or (axis == 1 and hrf.y < node.y) or (axis == 2 and hrf.z < node.z):
            return self.search_dfs(hrf, depth + 1, node.left)
        else:
            return self.search_dfs(hrf, depth + 1, node.right)

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
    
    def search_hash(self, hrf, node):
        # Catch base case of not finding a similar hash
        if node == None:
            node = self.root

        # Grab pertinent hash info
        hash_stage = 0
        while self.hash_stage < len(self.hash_order):
            hash_variable = hrf.hash_order[self.hash_stage]
            
            hrf.hashables = hrf.hash_context[hash_variable] 
            if isinstance(hrf.hashable, list) == False: hrf.hashables = [hrf.hashables]

            node.hashables = node.hash_context[hash_variable]
            if isinstance(hrf.hashable, list) == False: node.hashables = [node.hashables]

            for hrf.hashable in hrf.hashables:
                hrf_hash = hash(hrf.hashable)
                for node.hashable in node.hashables:
                    node_hash = hash(node.hashable)
                if hrf_hash == node_hash:
                    return node

            self.hash_stage += 1 # Check if we've reached base case

        node = self.search_hash(hrf, node.right) # Check the node to the right
            
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
    def __init__(self, trace = None, location = None, sfreq = None, trace_deviation = 'none', duration = 'none', stimulus = ['general'], conditions = ['general'], study = 'none', task = 'none', protocol = 'none', age_range = [0,99], demographics = 'general', doi = "", hrf_type = 'custom', plot = False, working_directory = None):
        # Set working directory and create plot 
        self.working_directory = working_directory or os.getcwd()
        if os.path.exists(f"{self.working_directory}/plots/") == False and plot:
            os.mkdir(f"{self.working_directory}/plots/")

        # Attach passed into info to class 
        if sfreq:
            self.sfreq = sfreq
        else:
            self.sfreq = 7.81

        if trace == None:
            self.trace = spm_hrf(self.sfreq)
        else:
            self.trace = trace

        if location:
            self.x = location[0]
            self.y = location[1]
            self.z = location[2]
        else:
            self.x = -1
            self.y = -1
            self.z = -1

        self.trace_deviation = trace_deviation

        self.type = hrf_type

        self.duration = duration
        self.stimulus = stimulus
        self.conditions = conditions
        self.study = study
        self.task = task
        self.protocol = protocol
        self.age_range = age_range
        self.demographics = demographics
        self.doi = doi

        self.hash_context = {
            'doi': self.doi,
            'task': self.task,
            'conditions': self.conditions,
            'stimulus': self.stimulus,
            'duration': self.duration,
            'protocol': self.protocol,
            'age_range': self.age_range,
            'demographics': self.demographics
        }

        self.left = None
        self.right = None


    def build(self, plot = False, show = False):
        # Define the processes for generating an hrf
        hrf_processes = [self.expand, self.compress, self.smooth, self.normalize]
        process_names = ['Expand', 'Compress', 'Smooth', 'Normalize']
        process_options = [None, self.mean_window, self.sigma, None]
        for process, process_name, process_option in zip(hrf_processes, process_names, process_options):
            
            if process_option == None:
                self.trace = process(self.trace)
            else:
                self.trace = process(self.trace, process_option)
            
            if plot: # Plot the processing step results
                self.plot(f"HRF - {process_name}", f"plots/{process_name.lower()}_{self.type}_hrf_results.png", show)

    def expand(self, hrf):
        # Continue to expand the hrf until it's bigger than size we need

        print('Expanding hrf...')
        while len(hrf) < self.length:
            # Define a new empty hrf to add in expanded hrf into
            new_hrf = [] 
            # Iterate through the current hrf
            for ind, data in enumerate(hrf): 
                # Append the current data point
                new_hrf.append(data) 
                # As long as theirs a datapoint in front to interpolate between
                if ind + 1 < len(hrf): 
                    # Interpolate a data point in between current datapoint and next
                    new_hrf.append((data + hrf[ind + 1])/2)
            hrf = new_hrf
        return hrf

    def compress(self, hrf):
        # Original list
        hrf_indices = np.linspace(0, len(hrf) - 1, len(hrf))

        # Create a spline interpolation function
        spline = interp1d(hrf_indices, hrf, kind='cubic')
        new_indices = np.linspace(0, len(hrf) - 1, int(self.length))

        # Compressed list
        return spline(new_indices)

    def smooth(self, hrf):
        # Smooth the hrf using a Gaussian blur
        print('Smoothing hrf with Gaussian hrf (sigma = {a})...')
        return hrf
        
    def normalize(self, hrf):
        hrf /= np.linalg.norm(hrf)
        return hrf
    
    def plot(self, title = None, filepath = None, show = True):
        plt.plot(self.trace)
        plt.title(title) 

        if filepath:
            plt.savefig(f'{self.working_directory}/plots/synthetic_hrf_base.jpeg')

        elif show:
            plt.show()
        
        plt.close()
        