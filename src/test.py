import getopt
import os
import signal
import skimage
from skimage.transform import resize
import sys

import h5py
import scipy.io as sio

signal.signal(signal.SIGINT, signal.SIG_DFL)
import time

import numpy as np
import utils as utl
os.environ['GLOG_minloglevel'] = '2'    # Supress most of caffe's log output
import caffe


class CaffePredictor:
    def __init__(self, prototxt, caffemodel, n_scales):       
        # Load a precomputed caffe model
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data_s0'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1)) # It's already RGB
        # Reshape net for the single input
        b_shape = self.net.blobs['data_s0'].data.shape
        self._n_scales = n_scales
        for s in range(n_scales):
            scale_name = 'data_s{}'.format(s)
            self.net.blobs[scale_name].reshape(b_shape[0],b_shape[1],b_shape[2],b_shape[3])

    # Probably it is not the eficient way to do it...
    def process(self, im, base_pw):
        # Compute dense positions where to extract patches
        [heith, width] = im.shape[0:2]
        pos = utl.get_dense_pos(heith, width, base_pw, stride=10)

        # Initialize density matrix and vouting count
        dens_map = np.zeros( (heith, width), dtype = np.float32 )   # Init density to 0
        count_map = np.zeros( (heith, width), dtype = np.int32 )     # Number of votes to divide
        
        # Iterate for all patches
        for ix, p in enumerate(pos):
            # Compute displacement from centers
            dx=dy=int(base_pw/2)
    
            # Get roi
            x,y=p
            sx=slice(x-dx,x+dx+1,None)
            sy=slice(y-dy,y+dy+1,None)
            crop_im=im[sx,sy,...]
            h, w = crop_im.shape[0:2]
            if h!=w or (h<=0):
                continue
            
            # Get all the scaled images
            im_scales = extractEscales([crop_im], self._n_scales)
            
            # Load and forward CNN
            for s in range(self._n_scales):
                data_name = 'data_s{}'.format(s)
                self.net.blobs[data_name].data[...] = self.transformer.preprocess('data', im_scales[0][s].copy())
            self.net.forward()
            
            # Take the output from the last layer
            # Access to the last layer of the net, second element of the tuple (layer, caffe obj)
            pred = list(self.net.blobs.items())[-1][1].data
            
            # Make it squared
            p_side = int(np.sqrt( len( pred.flatten() ) )) 
            pred = pred.reshape(  (p_side, p_side) )
            
            # Resize it back to the original size
            pred = utl.resizeDensityPatch(pred, crop_im.shape[0:2])          
            pred[pred<0] = 0

            # Sumup density map into density map and increase count of votes
            dens_map[sx,sy] += pred
            count_map[sx,sy] += 1

        # Remove Zeros
        count_map[ count_map == 0 ] = 1

        # Average density map
        dens_map = dens_map / count_map
        
        return dens_map


#===========================================================================
# Some helpers functions
#===========================================================================
# def testOnImg(CNN, im, gtdots, pw, mask = None):
def testOnImg(CNN, im, pw, mask = None):
    
    # Process Image
    resImg = CNN.process(im, pw) 

    # Mask image if provided
    if mask is not None:
        resImg = resImg * mask
        # gtdots = gtdots * mask

    npred=resImg.sum()
    # ntrue=gtdots.sum()

    # return ntrue,npred,resImg,gtdots
    return npred,resImg

def initTestFromCfg(cfg_file):
    '''
    @brief: initialize all parameter from the cfg file. 
    '''
    
    # Load cfg parameter from yaml file
    cfg = utl.cfgFromFile(cfg_file)
    
    # Fist load the dataset name
    dataset = cfg.DATASET
    
    # Set default values
    use_mask = cfg[dataset].USE_MASK
    use_perspective = cfg[dataset].USE_PERSPECTIVE
    
    # Mask pattern ending
    mask_file = cfg[dataset].MASK_FILE
        
    # Img patterns ending
    dot_ending = cfg[dataset].DOT_ENDING
    
    # Test vars
    test_names_file = cfg[dataset].TEST_LIST
    
    # Im folder
    im_folder = cfg[dataset].IM_FOLDER
    
    # Results output foder
    results_file = cfg[dataset].RESULTS_OUTPUT

    # Resize image
    resize_im = cfg[dataset].RESIZE

    # Patch parameters
    pw = cfg[dataset].PW # Patch with 
    sigmadots = cfg[dataset].SIG # Densities sigma
    n_scales = cfg[dataset].N_SCALES # Escales to extract
    perspective_path = cfg[dataset].PERSPECTIVE_MAP
    is_colored = cfg[dataset].COLOR

        
    return (dataset, use_mask, mask_file, test_names_file, im_folder, 
            dot_ending, pw, sigmadots, n_scales, perspective_path, 
            use_perspective, is_colored, results_file, resize_im)



def loadImage(filename, color=True):
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def extractEscales(lim, n_scales):
    out_list = []
    for im in lim:
        ph, pw = im.shape[0:2]  # get patch width and height
        scaled_im_list = []
        for s in range(n_scales):
            ch = s * ph // (2 * n_scales)
            cw = s * pw // (2 * n_scales)

            crop_im = im[ch:ph - ch, cw:pw - cw]

            scaled_im_list.append(resize(crop_im, (ph, pw)))

        out_list.append(scaled_im_list)

    return out_list


def main(argv, image_name):
    # Init parameters
    use_cpu = False
    gpu_dev = 0

    # Batch size
    b_size = -1

    # CNN vars
    prototxt_path = 'models/trancos/hydra2/hydra_deploy.prototxt'
    caffemodel_path = 'models/trancos/hydra2/trancos_hydra2.caffemodel'
        
        
    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["prototxt=", "caffemodel=", 
                                             "cpu_only", "dev=", "cfg="])
    except getopt.GetoptError as err:
        print("Error while parsing parameters: ", err)
        return
    
    for opt, arg in opts:
        if opt in ("--prototxt"):
            prototxt_path = arg
        elif opt in ("--caffemodel"):
            caffemodel_path = arg
        elif opt in ("--cpu_only"):
            use_cpu = True            
        elif opt in ("--dev"):
            gpu_dev = int(arg)
        elif opt in ("--cfg"):
            cfg_file = arg
            
    (dataset, use_mask, mask_file, test_names_file, im_folder,
            dot_ending, pw, sigmadots, n_scales, perspective_path, 
            use_perspective, is_colored, results_file, resize_im) = initTestFromCfg(cfg_file)

    # Set GPU CPU setting
    if use_cpu:
        caffe.set_mode_cpu()
    else:
        # Use GPU
        caffe.set_device(gpu_dev)
        caffe.set_mode_gpu()

    if use_perspective:
        pers_file = h5py.File(perspective_path,'r')
        pers_file.close()
        
    # Init CNN
    CNN = CaffePredictor(prototxt_path, caffemodel_path, n_scales)

    print("\nStart prediction for " + image_name)

    # Get image paths
    im_path = utl.extendName(image_name, im_folder)

    # Read image files
    im = loadImage(im_path, color = is_colored)

    if resize_im > 0:
        # Resize image
        im = utl.resizeMaxSize(im, resize_im)

    # Get mask if needed
    mask = None
    if use_mask:
        mask_im_path = utl.extendName(image_name, im_folder, use_ending=True, pattern=mask_file)
        mask = sio.loadmat(mask_im_path, chars_as_strings=1, matlab_compatible=1)
        mask = mask.get('BW')

    s=time.time()
    npred,resImg=testOnImg(CNN, im, pw, mask)
    print("image : %s, npred = %.2f , time =%.2f sec" % (image_name, npred, time.time() - s))

    return npred

if __name__=="__main__":
    main(sys.argv[1:], "Karlsruhe-Nord-B.jpg")