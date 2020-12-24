'''
David Bloomer (13161521)
Birkbeck University, MSc Data Science 18-20 PT
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.util import img_as_float64
from skimage.exposure import match_histograms, equalize_hist, equalize_adapthist, rescale_intensity, adjust_gamma, adjust_log
from skimage.color import rgb2gray, rgb2hsv, rgb2lab, hsv2rgb, label2rgb
from skimage.filters import gaussian, sobel, gabor, frangi, hessian, sato, meijering
from skimage.feature import canny, local_binary_pattern

from haralick import haralick_features

import os
from time import time
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# 3.3 Feature Engineering

def img_load(file, crop=False, x_offset=0, y_offset=0, crop_img_length=400):
    '''
    Load image data from specified file location (file), option to crop images
    on import.
    ----------------------
    Parameters
    ----------------------
    file: str, file location
    crop: Boolean (Default=True)
    x_offset: int (Default=0)
        x or column offset in pixels.
    y_offset: int (Default=0)
        y or row offset in pixels.
    crop_img_length: int (Default=400)
    ----------------------
    Returns
    ----------------------
    out: (N,M,3) ndarray
    '''
    assert os.path.isfile(file), 'Image not found.'
    img = imread(file)[:,:,:3]
    
    if crop:
        img = img[y_offset:y_offset+crop_img_length, x_offset:x_offset+crop_img_length, :]
    
    return img

def modify_clf_labels(clf):
    '''
    Modify classification labels on load. For iterative refinement of process.
    '''
    clf[clf == 6] = 5  # ilmenite and chromite -> opaques
    clf[clf == 7] = 6  # mafic_glass -> matrix
    
    clf[clf == 5] = 6  # merge opaques and matrix
    
    return clf

def clf_load(file, crop=False, x_offset=0, y_offset=0, crop_img_length=400):
    '''
    Load classification data from specified file location (file), option to crop
    classification on import. Labels may be manually overwritten by modification
    of the modify_clf_labels function.
    ----------------------
    Parameters
    ----------------------
    file: str, file location
    crop: Boolean (Default=True)
    x_offset: int (Default=0)
        x or column offset in pixels.
    y_offset: int (Default=0)
        y or row offset in pixels.
    crop_img_length: int (Default=400)
    ----------------------
    Returns
    ----------------------
    out: (N,M) ndarray
    '''
    assert os.path.isfile(file), 'Image not found.'
    clf = imread(file)
    
    if crop:
        clf = clf[y_offset:y_offset+crop_img_length, x_offset:x_offset+crop_img_length]

    # modify labels as required
    clf = modify_clf_labels(clf)
    
    # reassign unclassified label to be sequential (artefact of stores labels as greyscale images)
    clf[clf == 255] = 7
    
    return clf

def clf2rgb(clf):
    '''
    Convert classification labels to RGB. Modification to scjikit-image function
    label2rgb to overcome non-expected behavior where not all classes are present.
    Uses indexing of colormap to allow sequential allocation.
    ----------------------
    Parameters
    ----------------------
    clf: (N,M) ndarray
    ----------------------
    Returns
    ----------------------
    out: (N,M,3) ndarray
    '''
    colormap = np.array([[1.000, 1.000, 1.000],   # 0: background
                         [0.902, 0.471, 0.961],   # 1: pyroxene
                         [0.000, 0.502, 0.502],   # 2: plagioclase
                         [0.850, 0.850, 0.000],   # 3: olivine  # 0.745, 0.745, 0.000
                         [1.000, 0.753, 0.753],   # 4: silica
                         [0.000, 1.000, 0.000],   # 5: opaques
                         [0.400, 0.400, 0.400],   # 6: matrix  # 0.933, 0.910, 0.667
                         [0.000, 0.000, 0.000],   # 7: unknown
                         [0.498, 0.733, 0.733],   # 8: calcite (rubo)
                         [0.129, 0.098, 0.627],   # 9: dolomite (rubo)
                         [0.914, 0.945, 0.369]])  # 10: quartz (rubo)
    colors_present = np.unique(clf).astype(int)
    colormap = colormap[colors_present]  # index colormap using available labels
    
    return label2rgb(clf, colors=colormap)

def array_flatten(array):
    '''
    Conversion of 3- to 2-dimensional array, used to ingestion of image features
    within scikit-learn API classification functions.
    ----------------------
    Parameters
    ----------------------
    array: (N,M,d) ndarray
    ----------------------
    Returns
    ----------------------
    out: (N*M,d) ndarray
    '''    
    h, w, d = array.shape
    return array.reshape((h*w, d))

def array_restore(array, shape):
    '''
    Conversion scikit-learn API classification predictions to 2-d image data.
    ----------------------
    Parameters
    ----------------------
    array: (N*M,1) ndarray
    shape: tuple
    ----------------------
    Returns
    ----------------------
    out: (N,M) ndarray
    ''' 
    h, w = shape
    return array.reshape((h, w))

def feat_rgb(img):
    return img

def load_ml_file(folder, file_name, preproc=[], features=[feat_rgb], inc_xpl=True, auto_feat=False, crop=False, x_offset=0, y_offset=0, crop_img_length=400):
    '''
    For specified folder (folder) and image (file_name), load image and
    classification. Apply specified pre-processing (proproc) and features
    (features), and return data as 2-d array where classification labels
    occupy index position [:,-1]. Options to load ppl and xpl data, and crop
    image on import.
    Preprocessing and feature selection used in the final model can be loaded
    without paramterisation by setting auto_feat=True (requires inc_xpl=True).
    Expects images in format {file_name} following suffix:
        None: ppl
        a: xpl
        b: RGB classification (not used)
        c2: classification
    ----------------------
    Parameters
    ----------------------
    folder: directory path
    file_name: filename
    preproc: list (Default: [])
        List of pre-processing steps to apply to image prior to feature creation.
        Equalisation methods are not applied to xpl input.
    feature: list (Default: [feat_rgb])
        List of features to create ppl and/or xpl image(s).
    inc_xpl: Boolean (Default=True)
    auto_feat: Boolean (Default=False)
        Overrides preproc and feature parameters. Requires inc_xpl=True.
    crop: Boolean (Default=True)
    x_offset: int (Default=0)
        x or column offset in pixels.
    y_offset: int (Default=0)
        y or row offset in pixels.
    crop_img_length: int (Default=400)
    ----------------------
    Returns
    ----------------------
    out: (N*M,d) ndarray
    ''' 
    ppl = img_load(f'{folder}\\{file_name}.png', crop, x_offset, y_offset, crop_img_length)
    clf = clf_load(f'{folder}\\{file_name}c2.png', crop, x_offset, y_offset, crop_img_length)
    if inc_xpl:
        xpl = img_load(f'{folder}\\{file_name}a.png', crop, x_offset, y_offset, crop_img_length)
    
    # load only final feature set (more efficient)
    if auto_feat:
        # image pre-proccessing
        assert inc_xpl, 'XPL image must be included for pre-defined processing (inc_xpl=True).' 
        ppl, xpl = preproc_gaussian_filt(ppl), preproc_gaussian_filt(xpl)

        array = np.dstack([feat_rgb(ppl), feat_rgb(xpl), feat_hsv(ppl), feat_hsv(xpl),
                feat_la(ppl), feat_la(xpl), feat_sobel(ppl), feat_sobel(xpl),
                feat_gabor(ppl, ang_inc=4, frequencies=[0.1], grad=False),
                feat_gabor(xpl, ang_inc=1, frequencies=[0.1], grad=False),
                feat_frangi(ppl), feat_sato(ppl), feat_sato(xpl), feat_meijering(ppl),
                feat_meijering(xpl), feat_lbp4_uniform(ppl), feat_lbp8_uniform(ppl)])
        
    else:  
        # image pre-proccessing (inc. equalisation)
        for proc in preproc:
            ppl = proc(ppl)
            # only gaussian applied to xpl
            if inc_xpl and proc in [preproc_gaussian_filt]:
                xpl = proc(xpl)
        
        # build feature array
        for i, feat in enumerate(features):
            if i == 0:
                if inc_xpl:
                    array = np.dstack([feat(ppl), feat(xpl)])
                else:
                    array = feat(ppl)
            else:
                if inc_xpl:
                    array = np.dstack([array, feat(ppl), feat(xpl)])
                else:
                    array = np.dstack([array, feat(ppl)])
    
    array = np.dstack([array, clf])  # clf takes index -1
    
    # 2-dimensional array for use in training model
    return array_flatten(array)

def load_ml_dataset(folders, file_names, preproc=[], features=[feat_rgb], inc_xpl=True, auto_feat=False, max_files_per_folder=-1):
    '''
    For specified folder (folder) and images (file_names), sequentially load
    images and classifications. Apply specified pre-processing (proproc) and
    features (features), and return data as 2-d array where classification labels
    occupy index position [:,-1]. The number of images loaded from each folder
    is specified through max_files_per_folder, where -1 will load all files.
    Option to load ppl and xpl data.
    Preprocessing and feature selection used in the final model can be loaded
    without paramterisation by setting auto_feat=True (requires inc_xpl=True).
    Expects images in format {file_name} following suffix:
        None: ppl
        a: xpl
        b: RGB classification (not used)
        c2: classification
    ----------------------
    Parameters
    ----------------------
    folder: directory path
    file_name: filename
    preproc: list (Default: [])
        List of pre-processing steps to apply to image prior to feature creation.
        Equalisation methods are not applied to xpl input.
    feature: list (Default: [feat_rgb])
        List of features to create ppl and/or xpl image(s).
    inc_xpl: Boolean (Default=True)
    auto_feat: Boolean (Default=False)
        Overrides preproc and feature parameters. Requires inc_xpl=True.
    max_files_per_folder: int (Default=-1)
        Number of files to load from each folder. -1 for all files.
    ----------------------
    Returns
    ----------------------
    out: (N*M,d) ndarray
    ''' 
    for i, folder in enumerate(folders):
        if max_files_per_folder < 1:
            max_files_per_folder = len(file_names[i])
        for j, file in tqdm(enumerate(file_names[i][:max_files_per_folder]), desc=f'Sample {i}'):
            array_flatten = load_ml_file(folder, file, preproc, features, inc_xpl, auto_feat)
            
            if i+j == 0:  # only first iteration
                array_output = array_flatten
            else:
                array_output = np.vstack([array_output, array_flatten])
    
    return array_output

# Image Pre-Processing (see 3.1)
# wrapper functions for scikit-image functionality

def preproc_gaussian_filt(img, sigma=3):
    return gaussian(img, sigma=sigma, multichannel=True)

def eq_hist_match(img):
    ref_ppl = img_load(r'data\ref\ppl.png', crop=False)
    return np.uint8(match_histograms(img, ref_ppl, multichannel=True))

def eq_hist(img):
    _, _, d = img.shape
    return np.dstack([equalize_hist(img[:,:,i]) for i in range(d)])

def eq_hist_hsv(img):
    hsv = rgb2hsv(img)
    v = equalize_hist(hsv[:,:,2])
    return hsv2rgb(np.dstack([hsv[:,:,:2],v]))

def eq_adaptivehist(img):
    # Contrast Limited Adaptive Histogram Equalization (CLAHE), equalisation
    # performed in HSV value space.
    return equalize_adapthist(img)

def eq_rescale_intensity(img):
    return rescale_intensity(img, in_range=(0, 255))

def eq_adjust_gamma(img):
    _, _, d = img.shape
    return np.dstack([adjust_gamma(img[:,:,i]) for i in range(d)])

def eq_adjust_gamma_hsv(img):
    hsv = rgb2hsv(img)
    v = adjust_gamma(hsv[:,:,2])
    return hsv2rgb(np.dstack([hsv[:,:,:2],v]))

def eq_adjust_log(img):
    _, _, d = img.shape
    return np.dstack([adjust_log(img[:,:,i]) for i in range(d)])

def eq_adjust_log_hsv(img):
    hsv = rgb2hsv(img)
    v = adjust_log(hsv[:,:,2])
    return hsv2rgb(np.dstack([hsv[:,:,:2],v]))

# 3.3.1 Colour Representation
# wrapper functions for scikit-image functionality

def feat_greyscale(img):
    return rgb2gray(img)

def feat_hsv(img):
    return rgb2hsv(img)

def feat_lab(img):
    return rgb2lab(img)

def feat_la(img):
    return rgb2lab(img)[:,:,1:]

# 3.3.1 z-Normalised Colour Representation

def feat_norm_greyscale(img):
    gray = rgb2gray(img)
    return (gray - gray.mean()) / gray.std()

def feat_norm_rgb(img):
    img = img_as_float64(img)
    for c in range(3):
        img[:,:,c] = (img[:,:,c] - img[:,:,c].mean()) / img[:,:,c].std()
    return img

def feat_norm_hsv(img):
    hsv = rgb2hsv(img)
    for c in range(3):
        hsv[:,:,c] = (hsv[:,:,c] - hsv[:,:,c].mean()) / hsv[:,:,c].std()
    return hsv

def feat_norm_lab(img):
    lab = feat_lab(img)
    for c in range(3):
        lab[:,:,c] = (lab[:,:,c] - lab[:,:,c].mean()) / lab[:,:,c].std()
    return lab

# 3.3.2 Edge Filters
# wrapper functions for scikit-image functionality

def feat_sobel(img):
    img = feat_greyscale(img)
    return sobel(img)

def feat_canny(img, sigma=1):
    img = feat_greyscale(img)
    return canny(img, sigma=sigma)

# 3.3.3 Ridge Filters
# wrapper functions for scikit-image functionality

def feat_hessian(img):
    # hybrid hessian matrix (filter)
    img = feat_greyscale(img)
    return hessian(img)  # mode='reflect'

def feat_frangi(img):
    img = feat_greyscale(img)
    return frangi(img)

def feat_sato(img):
    img = feat_greyscale(img)
    return sato(img)  # mode='reflect'

def feat_meijering(img):
    img = feat_greyscale(img)
    return meijering(img)

# 3.3.4 Filter Banks

def calc_gradient(array):
    h, w, d = array.shape
    A = np.vstack([np.arange(d), np.ones(d)]).T
    y = array.T.reshape(d,-1)
    lin = np.linalg.lstsq(A, y, rcond=None)
    return np.reshape(lin[0][0,:], (w,h)).T

def feat_gabor(img, ang_inc=4, frequencies=[0.1,0.2,0.3,0.4], grad=True):
    '''
    Returns a Gabor Filter Bank at the specified angle ranges and frequencies as
    a n-dimensional array composed of each possible combination. Angles can be
    specified in the range of 0-3/4.pi radians (0-135 deg) at a 1/4.pi (45) step.
    Where grad=True, a least square regression is fit to Gabor products calculated
    at 0 radians and returned as the last-dimension in the output array.
    ----------------------
    Parameters
    ----------------------
    img: (N,M,3) ndarray
    ang_inc: int (Default=4)
    frequencies: list (Default=[0.1,0.2,0.3,0.4])
    grad: Boolean (Default=True)
    ----------------------
    Returns
    ----------------------
    out: (N,M,:) ndarray
    '''
    img = feat_greyscale(img)
    h, w = img.shape
    feat_map = np.zeros(shape=(h,w,ang_inc*len(frequencies)))
    for i in range(ang_inc):
        theta = i / 4. * np.pi  # eqv to (0,135,45) deg
        for j, freq in enumerate(frequencies): 
            d = (i*len(frequencies))+j  # dimension of feat_map to calculate
            filt_real, filt_imag = gabor(img, theta=theta, frequency=freq)
            feat_map[:,:,d] = (filt_real**2+filt_imag**2)**0.5
    if grad:
        # gradient calculated based on first angle
        grad = calc_gradient(feat_map[:,:,:len(frequencies)])
        feat_map = np.dstack([feat_map, grad])
    return feat_map

# 3.3.5 Neighbourhood Features
# wrapper functions for scikit-image functionality
# feat_haralick function is wrapper for code created by Jónathan Heras (Stack Overflow)

def feat_haralick(img, props=['contrast','dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']):
    img = feat_greyscale(img)
    feat_map = haralick_features(img, win=11, d=[1], theta=[0], levels=256, props=props)
    return feat_map

def feat_lbp1(img, r=1):
    # local binary pattern
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r)

def feat_lbp2(img, r=2):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r)

def feat_lbp3(img, r=3):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r)

def feat_lbp4(img, r=4):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r)

def feat_lbp8(img, r=8):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r)

def feat_lbp16(img, r=16):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r)

def feat_lbp2_uniform(img, r=2):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r, method='uniform')

def feat_lbp4_uniform(img, r=4):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r, method='uniform')

def feat_lbp8_uniform(img, r=8):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r, method='uniform')

def feat_lbp16_uniform(img, r=16):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r, method='uniform')

# Feature Evaluation and Reporting

def evaluate_feat_times(folders, features, plot=True):
    '''
    For reporting: Comparison of feature algorithm efficiency.
    '''
    global feature_times  # saved as global variable to retain progress on error
    feature_times = pd.DataFrame(columns=['feat','t_100','t_200','t_400','t_800','t_1600'])
    
    # build and evaluate feature array
    for i, (name, feat) in tqdm(enumerate(features), desc='Creating features'):
        times = []   
    
        for test_dim in [100, 200, 400, 800, 1600]:
            ppl = img_load(f'{folders[0]}\\1.png', crop=True, x_offset=300, y_offset=0, crop_img_length=test_dim)

            t1 = time()
            feat(ppl)
            times.append(time()-t1)
        
        feature_times.loc[i] = [name] + times
    
    if plot:
        plot_feat_times(feature_times)
        
    return feature_times

def plot_feat_times(dataframe='none'):
    '''
    For reporting: Visualisation of feature algorithm efficiency.
    '''
    if dataframe == 'none':
        dataframe = pd.DataFrame({'feat':['cielab (colorspace)','sobel (edge)','hessian (ridge)',
                                          'gabor (filter bank)','lbp (neighbourhood)','haralick (neighbourhood)'],
                                   't_100':[0.00, 0.00, 0.01, 0.01, 0.10, 35.6],
                                   't_200':[0.01, 0.01, 0.03, 0.02, 0.22, 119.4],
                                   't_400':[0.07, 0.02, 0.37, 0.08, 0.77, 465.7],
                                   't_800':[0.30, 0.04, 1.61, 0.32, 2.79, 1908.2],
                                   't_1600':[1.12, 0.17, 6.11, 1.35, 10.58, 7246.8]})

    dataframe = dataframe.set_index('feat').T.set_index(pd.Index([100**2, 200**2, 400**2, 800**2, 1600**2]))
    for col in dataframe.columns:
        dataframe[col] = dataframe[col] / 60
    ax = dataframe.plot(kind='line')
    ax.set_xlabel('Pixel count')
    ax.set_ylabel('Time (mins)')

def plot_equalisation(folders):
    '''
    For reporting: Visualisation of CLAHE image equalisation.
    '''
    test_dim = 400
    test_images = [(folders[0], '1', 1300, 700),
                   (folders[1], '10', 300, 1500),
                   (folders[2], '12', 200, 100),
                   (folders[3], '1', 1000, 1100)]
    
    fig = plt.figure(figsize=(8, 8))
    
    for i, (folder, file_name, x, y) in enumerate(test_images):
        ppl = img_load(f'{folder}\\{file_name}.png', crop=True, x_offset=x, y_offset=y, crop_img_length=test_dim)
        xpl = img_load(f'{folder}\\{file_name}a.png', crop=True, x_offset=x, y_offset=y, crop_img_length=test_dim)
        clf = clf2rgb(clf_load(f'{folder}\\{file_name}c2.png', crop=True, x_offset=x, y_offset=y, crop_img_length=test_dim))
        ppl_eq = eq_adaptivehist(ppl)
        xpl_eq = eq_adaptivehist(xpl)
        
        img_to_plot = [ppl, xpl, clf, ppl_eq, xpl_eq]
        
        for j, img in enumerate(img_to_plot):  # 
            ax = fig.add_subplot(len(test_images), len(img_to_plot), (i*len(img_to_plot))+(j+1))
            plt.imshow(img)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    
    plt.show()

#%%############################################################################

def main():
    '''
    Pseudo-code to demonstrate applied workflow.
    '''
    # define data folders per sample
    folders = [r'data\15125,6',
               r'data\15475,15',
               r'data\15555,209',
               r'data\15597,18']
    file_names = [('1','2','3','4'),
                  ('1','2','6','10','11','12','14'),
                  ('1','7','8','12'),
                  ('1','2','3','4')]
    
    # feature algorithm efficiency                                    
    feature_classes =  [('cielab (colorspace)', feat_lab),
                        ('sobel (edge)', feat_sobel),
                        ('hessian (ridge)', feat_hessian),
                        ('gabor (filter bank)', feat_gabor),
                        ('lbp (neighbourhood)', feat_lbp16),
                        ('haralick (neighbourhood)', feat_haralick)                    
                        ]
    evaluate_feat_times(folders, feature_classes)
    
    # image equalisation visualisation
    plot_equalisation(folders)

    # load data sample
    sample = load_ml_dataset(folders,
                             file_names,
                             preproc=[preproc_gaussian_filt],
                             features=[feat_rgb],
                             inc_xpl=True,
                             max_files_per_folder=1)

