### Requirements
1. Use a Linux distribution
2. [Caffe](http://caffe.berkeleyvision.org/) and pycaffe

**Note:** If you build Caffe from source, it must be built with support for Python layers!

```
# In your Makefile.config, make sure to have this line uncommented
WITH_PYTHON_LAYER := 1
```

#### Python 3 Modules
- bottle
- CherryPy
- urllib3
- scipy
- scikit-image
- numpy
- h5py
- opencv-python
- easydict
- Pillow
- Cython

### Run
Comment line 20 and uncomment line 22 in ```run.sh``` if you are using Caffe-CPU instead of Caffe-Cuda.

Execute ```run.sh```  
The Server is listening for HTTP-GET requests on port 8079.