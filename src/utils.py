import os

import numpy as np
from skimage.transform import resize


def resizeMaxSize(im, max_size):
    """
    Resize an image leting its maximun size to max_size without modifing its 
    aspect ratio.
    @param im: input image
    @param max_size: maxinum size
    @return: resized image
    """

    h,w = im.shape[0:2]
    
    im_resized = None
    if w > h:
        s = float(max_size)/w
        im_resized = resize(im, (int(h*s), max_size))
    else:
        s = float(max_size)/h
        im_resized = resize(im, (max_size, int(w*s)))
    
    return im_resized    

def cfgFromFile(filename):
    """Load a config file."""
    import yaml
    from easydict import EasyDict as edict
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    return yaml_cfg

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def get_dense_pos(heith, width, pw, stride = 1):
    '''
    @brief: Generate a dense list of patch position.
    @param heith: image height.
    @param width: image width.
    @param pw: patch with.
    @param stride: stride.
    @return: returns a list with the patches positions.
    '''    
    # Compute patch halfs
    dx=dy=pw//2
    # Create a combination which corresponds to all the points of a dense
    # extraction
    pos = cartesian( (list(range(dx, heith - dx, stride)), list(range(dy, width -dy, stride)) ) )
#    return pos
    bot_line = cartesian( (heith - dx -1, list(range(dy, width -dy, stride)) ) )
    right_line = cartesian( (list(range(dx, heith - dx, stride)), width -dy - 1) )
    return np.vstack( (pos, bot_line, right_line) )

def resizeDensityPatch(patch, opt_size):
    '''
    @brief: Take a density map and resize it to the opt_size.
    @param patch: input density map.
    @param opt_size: output size.
    @return: returns resized version of the density map.    
    '''
    # Get patch size
    h, w = patch.shape[0:2]
    
    # Total sum
    patch_sum = patch.sum()

    # Normalize values between 0 and 1. It is in order to performa a resize.    
    p_max = patch.max()
    p_min = patch.min()
    # Avoid 0 division
    if patch_sum !=0:
        patch = (patch - p_min)/(p_max - p_min)
    
    # Resize
    patch = resize(patch, opt_size)
    
    # Return back to the previous scale
    patch = patch*(p_max - p_min) + p_min
    
    # Keep count
    res_sum = patch.sum()
    if res_sum != 0:
        return patch * (patch_sum/res_sum)

    return patch

def extendName(name, im_folder, use_ending=False, pattern=[]):
    '''
        @brief: This gets a file name format and adds the root directory and change 
        the extension if needed
        @param fname: file name.
        @param im_folder: im_folder path to add to each file name.
        @param use_ending: flag use to change the file extension.
        @param pattern: string that will substitute the original file ending. 
        @return new_name: list which contains all the converted names.
    '''    
    final_name = im_folder + os.path.sep + name
    
    if use_ending:
        l_dot = final_name.rfind('.')
        final_name = final_name[0:l_dot] + pattern
    
    return final_name