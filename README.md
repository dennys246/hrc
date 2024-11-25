# hrc
This repository contains the hemodynamic response convolver (hrc) tool which can be used on preprocessed mne nirx objects to extract the hemodynamic response function. Within this library also contains the HRF object that can be used to dynamically generate or load synthetic HRF filters for use in convolution. 

To use this repository all you'll need to do is install the tool via pip...

`pip install hrconv`

From there all you need to do is import the library in a python script and pass in your MNE NIRS objects to the main hrconv.convolve_hrf(scan) function

```
import hrconv as hrc

conv_nirx_obj = hrc.convolve_hrf(nirx_obj)
```

you can also create your own unique HRF's for convolving your NIRS data with by calling to hrconv.hrf class with your own unique parameters...

```

hrf = hrc.hrf(filter_type = 'double-gamma')

conv_nirx_obj = hrc.convolve_hrf(nirx_obj, hrf)

```
