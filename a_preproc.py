'''
David Bloomer (13161521)
Birkbeck University, MSc Data Science 18-20 PT

References:
https://stackoverflow.com/questions/33196130/replacing-rgb-values-in-numpy-array-by-integer-is-extremely-slow
https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
'''

import numpy as np
import cv2 as cv

import os
from tqdm import tqdm

# 3.1 Image Pre-processing (classification labeling, noise removal)

def bgr2label(img, lookup_dict, default=255):
    '''
    Convert bgr values within input image (img) to classification labels based
    on entries stored within user defined mineral dictionary (lookup_dict).
    Where BGR not contained within dictionary, the default classification label
    (default) is used. A default of 255 is used to give a classification range
    of 0-255, and allow classifications to be stored as greyscale images.
    ----------------------
    Parameters
    ----------------------
    img: (N,M,3) ndarray
    lookup_dict: user defined mineral dictionary
    default: int (Default=255)
    ----------------------
    Returns
    ----------------------
    out: (N,M) ndarray
    '''
    clf = np.ndarray(shape=img.shape[:2], dtype=np.uint8)
    clf[:,:] = default  # unclassified=255
    
    for gbr, idx in tqdm(lookup_dict.items()):
        clf[(img[:,:,:3]==gbr).all(2)] = idx
    
    return clf

def remove_noise_connected(clf, min_thesh=40, new_class=0):
    '''
    Remove noise by setting connected components below minimum theshold
    (min_thesh) for classification (clf) to specified label (new_class).
    Supports binary or multi-label classifiers.
    ----------------------
    Parameters
    ----------------------
    clf: (N,M) ndarray
    min_thesh: int (Default=40)
    new_class: int (Default=0)
    ----------------------
    Returns
    ----------------------
    out: (N,M) ndarray
    '''
    clf_denoise = clf.copy()

    for i in np.unique(clf)[1:]:
        array = np.zeros(clf.shape, dtype=np.uint8)
        array[clf == i] = 1
        n_comp, labels, stats, centroids = cv.connectedComponentsWithStats(array, connectivity=8)
        sizes = stats[1:, -1]
        
        for j in tqdm(range(0, n_comp-1), desc=f'Class {i}'):  # ignore background (largest connected component)
            if sizes[j] <= min_thesh:
                clf_denoise[labels == j+1] = new_class
    
    return clf_denoise

def remove_noise_morph_open(clf):
    '''
    Remove noise within classification (clf) through application of morphological
    erosion and dilation operators.
    Alternative approach to using remove_noise_connected() and significantly faster.
    ----------------------
    Parameters
    ----------------------
    clf: (N,M) ndarray
    ----------------------
    Returns
    ----------------------
    out: (N,M) ndarray
    '''
    kernel = np.ones((5,5), np.uint8)
    clf_denoise = cv.morphologyEx(clf, cv.MORPH_OPEN, kernel)
    
    return clf_denoise

def remove_noise_morph_close(clf, inc=2, apply_class=0):
    '''
    Remove noise within classification (clf) through application of morphological
    dilation and erosion operators. Operators are applied for number of iterations
    (inc) where operator increases by one-pixel per iteration to minimise operator
    artefacts. Operation is applied only to the background label (apply_class=0).
    ----------------------
    Parameters
    ----------------------
    clf: (N,M) ndarray
    inc: int (Default=2)
    apply_class: int (Default=0)
    ----------------------
    Returns
    ----------------------
    out: (N,M) ndarray
    '''
    array = clf.copy()
    k_size = 2
    for _ in range(inc):
        kernel = np.ones((k_size,k_size), np.uint8)
        array = cv.morphologyEx(array, cv.MORPH_CLOSE, kernel)
        k_size += 1
    
    clf_denoise = np.where(clf == apply_class, array, clf)
    
    return clf_denoise

def create_lut(lookup_dict, contrast, default):
    '''
    Create lut [opencv lookup table] from user defined mineral dictionary (lookup_dict)
    for use within clf2bgr(). For iterative refinement of process.
    Relationship between colour and mineral label can be ignore to create a high
    contrast image (contrast=True), specify the background colour (default) as
    either black (0) or white (255).
    ----------------------
    Parameters
    ----------------------
    lookup_dict: user defined mineral dictionary
    contrast: Boolean
    default: int [0, 255]
    ----------------------
    Returns
    ----------------------
    out: (1,256,3) ndarray
    '''
    lut = np.zeros((1,256,3), np.uint8)
    lut[0,0] = [default,default,default]
    
    # for high contrast interpretation, all other classes contrast with default
    if contrast:
        assert default in [0, 255], 'default must be either 0 or 255.'
        g = abs(default-255)
        lut[0,1:] = [g,g,g]
        return lut
    
    # iterate through dictionary and build lut
    for i in range(1,9):
        try:
            b, g, r = [key for key, value in lookup_dict.items() if value == i][0]
            lut[0,i] = [b,g,r]
        except IndexError:
            pass
    return lut

def clf2bgr(clf, lookup_dict, contrast=False, default=255, trans=False):
    '''
    Convert classification labels to BGR as specified through user defined
    mineral dictionary (lookup_dict) image. opencv implementation of scikit-image
    label2rgb.
    Relationship between colour and mineral label can be ignore to create a high
    contrast image (contrast=True), specify the background colour (default) as
    either black (0) or white (255).
    To calculate alpha transparacy mask based on background label set trans=True.
    ----------------------
    Parameters
    ----------------------
    clf: (N,M) ndarray
    lookup_dict: user defined mineral dictionary
    contrast: Boolean (Default=False)
    default: int (Default=255)
    trans: Boolean (Default=False)
        Create alpha transparacy mask, increase output dimensions to 4.
    ----------------------
    Returns
    ----------------------
    out: (N,M,3/4) ndarray
    '''
    assert clf.dtype == np.uint8, 'Input must have dtype uint8.'
    
    lut = create_lut(lookup_dict, contrast=contrast, default=default)
    
    label_bgr = cv.LUT(cv.merge((clf, clf, clf)), lut)

    # create transparency mask and append
    if trans:
        mask = np.where(clf==0, 0, 255)
        label_bgr = np.dstack([label_bgr, mask])

    return label_bgr

def create_labels(folder, ref, label_index, contrast_index, remove_noise=True):
    '''
    For specified folder (folder) and reference image (ref), create sequential
    integer label representation, as specified by user defined mineral dictionary
    (label_index), and high contrast classification for use in image alignment,
    as specified by user defined mineral dictionary (contrast_index). Noise may
    be optionally removed from the high contrast classification. 
    ----------------------
    Parameters
    ----------------------
    folder: directory path
    ref: filename
    label_index: user defined mineral dictionary
    contrast_index: user defined mineral dictionary
    remove_noise: Boolean (Default=True)
    ----------------------
    Returns
    ----------------------
    clf: (N,M1) ndarray
    clf_bgr: (N,M,3) ndarray
    clf_contrast: (N,M) ndarray
    '''
    # load data (reference)
    in_ref = rf'{folder}\inputs\{ref}'
    ref = cv.imread(in_ref, cv.IMREAD_COLOR)
    
    assert ref.dtype == np.uint8, 'Reference image failed to load.'
    
    # output file locations
    out_clf = rf'{folder}\inputs\label.png'
    out_interp = rf'{folder}\inputs\interp.png'
    out_contrast = rf'{folder}\inputs\contrast.png'

    # create labels
    print('\nCreating classification labels..')
    clf = bgr2label(ref, label_index, default=255)
    print('\nCreating high contrast interpretation.')
    clf_contrast = bgr2label(ref, contrast_index, default=0)
    
    # remove noise (if required)
    if remove_noise:
        print('Removing noise from high contrast interpretation.')
        clf_contrast = remove_noise_connected(clf_contrast)
    
    # convert label to BGR
    clf_bgr = clf2bgr(clf, label_index, trans=False)
    
    # save outputs
    cv.imwrite(out_clf, clf)
    cv.imwrite(out_interp, clf_bgr)
    cv.imwrite(out_contrast, clf_contrast)
    print('Process complete.')
    
    return clf, clf_bgr, clf_contrast

def clf_denoise(folder, clf, label_index, min_thesh=6, inc=2):
    '''
    For specified folder, remove noise from classification (clf). Connected
    components under minimum threshold (min_thesh) are replaced with the background
    label. Morphological erosion and dilation is then applied to anneal gaps
    within the background label. Operators are applied for number of iterations
    (inc) where operator increases by one-pixel per iteration to minimise operator
    artefacts.
    ----------------------
    Parameters
    ----------------------
    folder: directory path
    clf: filename
    label_index: user defined mineral dictionary
    min_thesh: int (Default=6)
    inc: int (Default=2)
    ----------------------
    Returns
    ----------------------
    clf: (N,M) ndarray
    clf_bgr: (N,M,3) ndarray
    '''
    # Load data (reference) 
    file_name = clf.split('.')[0]
    clf = f'{folder}\inputs\{clf}'
    
    assert os.path.isfile(clf), f'{clf} not found.'

    clf = cv.imread(clf, cv.IMREAD_UNCHANGED)
    
    # Output file locations
    out_clf = rf'{folder}\inputs\{file_name}1.png'
    out_clf_bgr = rf'{folder}\inputs\{file_name}d.png'
    

    clf = remove_noise_connected(clf, min_thesh=min_thesh)
    clf = remove_noise_morph_close(clf, inc=inc)
    clf_brg = clf2bgr(clf, label_index, trans=False)
    
    cv.imwrite(out_clf, clf)
    cv.imwrite(out_clf_bgr, clf_brg)
    print('Process complete.')
    
    return clf, clf_brg

# 3.2 Image Alignment

def border_add(img):
    '''
    Add 50px border to image (img). To prevent data loss through minor
    misalignment in initial alignment.
    ----------------------
    Parameters
    ----------------------
    img: (N,M,3) ndarray
    ----------------------
    Returns
    ----------------------
    out: (N,M,3) ndarray
    '''
    return cv.copyMakeBorder(img, 50, 50, 50, 50, borderType=cv.BORDER_CONSTANT, value=(255,255,255))

def border_remove(img, clf):
    '''
    Crop image (img) and classification (clf) to remove any rows or columns
    where the classification is composed entirely of whitespace.
    ----------------------
    Parameters
    ----------------------
    img: (N,M,3) ndarray
    clf: (N,M) ndarray
    ----------------------
    Returns
    ----------------------
    img: (N,M,3) ndarray
    clf: (N,M) ndarray
    '''
    assert clf.dtype == np.uint8, 'Input must have dtype uint8.'
    
    clf = clf[50:-50, 50:-50]  # remove clf padding
    
    assert (img.shape[0] == clf.shape[0]) & (img.shape[1] == clf.shape[1]), 'Images are not same size.'
    
    idx = cv.findNonZero(clf)
    x, y, w, h = cv.boundingRect(idx)
    
    return img[y:y+h, x:x+w, :], clf[y:y+h, x:x+w]

def align_preproc(img, grad_type='none', gaussian_filter=5):
    '''
    Perform image (img) pre-processing to improve image alignment.
    Default pre-processing (grad_type='none') performs colour space conversion
    to greyscale. Otherwise image is converted to image gradient though application
    of Sobel, Scharr or Laplacian operators. As gradient sensitive to noise,
    an optional Gaussian filter is applied prior to calculation.
    ----------------------
    Parameters
    ----------------------
    img: (N,M,3) ndarray
    grad_type: {'none','sobel','scharr','laplacian'} (Default:'none')
    gaussian_filter: int (Default=5)
        Gaussian filter applied prior to gradient calculation, if grad_type != 'none'.
    ----------------------
    Returns
    ----------------------
    out: (N,M,3) ndarray
    '''
    # convert image to greyscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if grad_type != 'none':
        # Gaussian filter
        img = cv.GaussianBlur(img, (gaussian_filter, gaussian_filter), 0)
        # Sobel gradient
        if grad_type == 'sobel':
            img = cv.addWeighted(np.absolute(cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)), 0.5,
                                 np.absolute(cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)), 0.5, 0)
        # Scharr gradient
        elif grad_type == 'scharr':
            img = cv.addWeighted(np.absolute(cv.Scharr(img, cv.CV_32F, 1, 0, scale=1)), 0.5,
                                 np.absolute(cv.Scharr(img, cv.CV_32F, 0, 1, scale=1)), 0.5, 0)
        # Laplacian gradient
        elif grad_type == 'laplacian':
            img = np.absolute(cv.Laplacian(img, cv.CV_32F, ksize=1))
        
        # convert image back to uint8 (gradient calculated in 32F to avoid loss of negative gradients)
        img = np.uint8(img)
    
    return img

def align_clf(img, contrast, clf, preproc='gs', finder='sift', matcher='flann', lowe_ratio=True, match_conf=0.9, warp_type='affine-rigid'):
    '''
    Rescale and align classification (clf) to image (img). Feature matching and
    alignment is performed between high contrast classification (contrast) and image.
    ----------------------
    Parameters
    ----------------------
    img: (N,M,3) ndarray
    contrast: (N,M) ndarray
    clf: (N,M) ndarray
    preproc: {'gs','sobel','scharr','laplacian'} (Default='gs')
        Optional image-processing applied to image and high contrast classification
        prior to feature extraction.
    finder: {'sift','orb'} (Default='sift')
        Feature extraction method.
    matcher: {'flann','bruteforce'}
        Feature matching method.
    lowe_ratio: Boolean (Default=True)
        Lowe ratio test. If not performed, matches are ordered by rank and top
        fraction used as specied by match_conf.
    match_conf: float (Default: 0.9)
    warp_type: {'affine-rigid','affine','homography'} (Default: 'affine-rigid')
        Homography matrix applied to perform warp.
    ----------------------
    Returns
    ----------------------
    img_matches: (N,M,3) ndarray
        Visualisation of feature matches.
    contrast_warp: (N,M) ndarray
        Warp applied to high contrast classification (contrast).
    clf_warp: (N,M) ndarray
        Warp applied to classification (clf)
    '''
    # image pre-processing
    if preproc == 'gs':
        img_preproc, contrast_preproc = align_preproc(img), align_preproc(contrast)
    elif preproc == 'sobel':
        img_preproc, contrast_preproc = align_preproc(img, grad_type='sobel'), align_preproc(contrast, grad_type='sobel')
    elif preproc == 'scharr':
        img_preproc, contrast_preproc = align_preproc(img, grad_type='scharr'), align_preproc(contrast, grad_type='scharr')
    elif preproc == 'laplacian':
        img_preproc, contrast_preproc = align_preproc(img, grad_type='laplacian'), align_preproc(contrast, grad_type='laplacian')
        
    # define matcher and required parameters
    if finder == 'orb':
        finder = cv.ORB_create(scoreType=0)  # Harris corner points
        if matcher == 'flann':
            # FLANN algorithm configuration for ORB/BRISK
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6,  # 12 (default docs)
                               key_size = 12,  # 20
                               multi_probe_level = 1)  #2
        else:
            norm_type = cv.NORM_HAMMING
    else:
        if finder == 'sift':
            finder = cv.xfeatures2d.SIFT_create()
        elif finder == 'surf':
            finder = cv.xfeatures2d.SURF_create()
        if matcher == 'flann':
            # FLANN algorithm configuration for SIFT/SURF
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            norm_type = cv.NORM_L2

    # detect features and compute descriptors
    kp1, des1 = finder.detectAndCompute(img_preproc, None)
    kp2, des2 = finder.detectAndCompute(contrast_preproc, None)

    if matcher == 'bruteforce':
        if lowe_ratio:
            matcher = cv.BFMatcher(normType=norm_type)
        else:
            matcher = cv.BFMatcher(normType=norm_type, crossCheck=True)  # crossCheck provides alternative to Lowe ratio test            
    elif matcher == 'flann':
        # Create matcher object
        matcher = cv.FlannBasedMatcher(index_params, dict(checks=100))
    
    if lowe_ratio:
        # match descriptors
        matches = matcher.knnMatch(des1, des2, k=2)
        # Lowe raio test
        matches_qc = []
        matches_qc_plt = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                matches_qc.append(m)
                matches_qc_plt.append([m])  # drawMatchesKnn() requires list of lists
        # draw matches
        img_matches = cv.drawMatchesKnn(img_preproc, kp1, contrast_preproc, kp2, matches_qc_plt, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        matches = matcher.match(des1, des2)
        # sort in order of distance, keep highest fract of matches
        matches = sorted(matches, key = lambda x:x.distance)
        matches_qc = matches[:int(len(matches)*(1-match_conf))]
        img_matches = cv.drawMatches(img_preproc, kp1, contrast_preproc, kp2, matches_qc, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # reshape matches to source-destination
    assert len(matches_qc) > 3, 'Not enough features detected.'
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches_qc]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches_qc]).reshape(-1,1,2)

    # estimate homography and apply warp to image(s)
    height, width, channels = img.shape
    bg_colour = (0,0,0)  # override background colour of rotated image: default black

    if warp_type == 'affine-rigid':  # rotation, translation and scale
        M, mask = cv.estimateAffinePartial2D(src_pts, dst_pts)
        contrast_warp = cv.warpAffine(contrast, M, (width, height), flags=cv.WARP_INVERSE_MAP, borderValue=bg_colour)
        clf_warp = cv.warpAffine(clf, M, (width, height), flags=cv.WARP_INVERSE_MAP, borderValue=bg_colour)
    elif warp_type == 'affine':  # rotation, translation, scale and shear
        M, mask = cv.estimateAffine2D(src_pts, dst_pts)
        contrast_warp = cv.warpAffine(contrast, M, (width, height), flags=cv.WARP_INVERSE_MAP, borderValue=bg_colour)
        clf_warp = cv.warpAffine(clf, M, (width, height), flags=cv.WARP_INVERSE_MAP, borderValue=bg_colour)
    elif warp_type == 'projective':  # three-dimensional translation (photography)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        contrast_warp = cv.warpPerspective(contrast, M, (width, height), flags=cv.WARP_INVERSE_MAP, borderValue=bg_colour)
        clf_warp = cv.warpPerspective(clf, M, (width, height), flags=cv.WARP_INVERSE_MAP, borderValue=bg_colour)

    return img_matches, contrast_warp, clf_warp

def create_aligned_inputs(folder, img, clf_contrast, clf, label_index, one_pass_only=False, save_all=False):
    '''
    For specified folder (folder) and reference image (ref), rescale and align
    classification (clf) to image (img). Feature matching and alignment is
    performed between high contrast classification (contrast) and image.
    By default, the function performs a two-pass alignment, using a affine-rigid
    then affine homography warp. Outputs are cropped to remove rows or columns
    composed entirely of the background label.
    ----------------------
    Parameters
    ----------------------
    folder: directory path
    ref: filename
    img: (N,M,3) ndarray
    clf_contrast: (N,M) ndarray
    clf: (N,M) ndarray
    label_index: user defined mineral dictionary
    one_pass_only: Boolean (Default=False)
        If True, second homography warp not applied.
    save_all: Boolean (Default=False)
        If True, intermediate steps (first-pass warp, visualisation of feature
        matches) are output to disk for use in reporting.
    ----------------------
    Returns
    ----------------------
    img_final: (N,M,3) ndarray
    clf_final: (N,M) ndarray
    clf_brg_final: (N,M,3) ndarray
    '''
    # load data
    file_name = img.split('.')[0]
    img = rf'{folder}\inputs\{img}'
    clf_contrast = rf'{folder}\inputs\{clf_contrast}'
    clf = rf'{folder}\inputs\{clf}'

    for file_loc in [img, clf_contrast, clf]:
        assert os.path.isfile(file_loc), f'{file_loc} not found.'
    
    img = cv.imread(img, cv.IMREAD_COLOR)
    clf_contrast = cv.imread(clf_contrast, cv.IMREAD_COLOR)  # high contrast classification
    clf = cv.imread(clf, cv.IMREAD_UNCHANGED)  # classification

    # output file locations
    out_match1 = rf'{folder}\{file_name}_match_warp1.png'
    out_contrast_warp1 = rf'{folder}\{file_name}_contrast_warp1.png'
    out_match2 = rf'{folder}\{file_name}_match_warp2.png'
    out_contrast_warp2 = f'{folder}\{file_name}_contrast_warp2.png'
    out_img = rf'{folder}\{file_name}a.png'
    out_clf_class_bgr = rf'{folder}\{file_name}b.png'
    out_clf_class = rf'{folder}\{file_name}c.png'
    
    # add border
    img_b = border_add(img)
    
    # perform affine-rigid alignment
    print('Performing alignment 1 (affine-rigid)..')
    img_match1, clf_contrast_warp1, clf_warp1 = align_clf(img_b,
                                            clf_contrast,
                                            clf,
                                            preproc='gs',
                                            finder='sift',
                                            matcher='flann',
                                            lowe_ratio=True,
                                            match_conf=0.9,
                                            warp_type='affine-rigid')
    
    # output intermediate processing steps
    if save_all:
        cv.imwrite(out_match1, img_match1)
        cv.imwrite(out_contrast_warp1, clf_contrast_warp1)

    # perform affine alignment
    if one_pass_only:
        # remove whitespace
        img_final, clf_final = border_remove(img, clf_warp1)
    else:
        print('Performing alignment 2 (affine)..')
        img_match2, clf_contrast_warp2, clf_warp2 = align_clf(img_b,
                                            clf_contrast_warp1,
                                            clf_warp1,
                                            preproc='gs',
                                            finder='sift',
                                            matcher='flann',
                                            lowe_ratio=True,
                                            match_conf=0.9,
                                            warp_type='affine')
        
        # remove whitespace
        img_final, clf_final = border_remove(img, clf_warp2)
    
    # convert label to bgr (option to include transparancy mask)
    clf_brg_final = clf2bgr(clf_final, label_index, trans=False)
    
    # save outputs
    cv.imwrite(out_img, img_final)
    cv.imwrite(out_clf_class, clf_final)
    cv.imwrite(out_clf_class_bgr, clf_brg_final)
    print('Process complete.')
    
    # output intermediate processing steps
    if save_all and not one_pass_only:
        cv.imwrite(out_match2, img_match2)
        cv.imwrite(out_contrast_warp2, clf_contrast_warp2)
        
    return img_final, clf_final, clf_brg_final

def align_ppl_xpl(folder, img1, img2):
    '''
    For specified folder (folder), perform single pass alignment of image (img2)
    against reference image (img1). For use in alignment of PPL and XPL microscopy
    images.
    ----------------------
    Parameters
    ----------------------
    folder: directory path
    img1: filename
        Reference image.
    img2: filename
    ----------------------
    Returns
    ----------------------
    out: (N,M,3) ndarray
    '''
    # load data
    file_name = img1.split('.')[0]
    img1 = rf'{folder}\{img1}'
    img2 = rf'{folder}\{img2}'
    
    for file_loc in [img1, img2]:
        assert os.path.isfile(file_loc), f'{file_loc} not found.'
    
    img1 = cv.imread(img1, cv.IMREAD_COLOR)
    img2 = cv.imread(img2, cv.IMREAD_COLOR)

    # output file locations
    out_img2 = rf'{folder}\{file_name}al.png'
    
    # perform affine-rigid alignment
    print('Performing alignment..')
    _, img2_warp1, _ = align_clf(img1,
                                 img2,
                                 img2,  # align image twice (workaround)
                                 preproc='gs',
                                 finder='sift',
                                 matcher='flann',
                                 lowe_ratio=True,
                                 match_conf=0.9,
                                 warp_type='affine-rigid')
    
    # save outputs
    cv.imwrite(out_img2, img2_warp1)
    print('Process complete.')
        
    return img2_warp1

#%%############################################################################

def main():
    '''
    Pseudo-code to demonstrate applied workflow.
    '''
    # define dictionary to group minerals
    # classification labels and high contrast maps
    label_index = {
        # (bgr) -> group_id, unclassified minerals not defined
        # 1: Pyroxene
        (245,120,230):  1,  # augite
        (223,155,180):  1,  # hedenbergite
        (235,15,210):   1,  # pigeonite
        (245,35,149):   1,  # clinoenstatite
        # 2: Plagioclase Feldspar
        (128,128,0):    2,  # plagioclase_an90
        (175,175,0):    2,  # plagioclase_an90
        (244,244,0):    2,  # plagioclase_an70
        # 3: Olivine
        (0,190,190):    3,  # olivine_fo80
        (0,170,170):    3,  # olivine_fo70
        (0,150,150):    3,  # olivine_fo60
        (0,130,130):    3,  # olivine_fo50
        (0,110,110):    3,  # olivine_fo40
        (0,90,90):      3,  # olivine_fo30
        (0,70,70):      3,  # olivine_fo20
        (0,50,50):      3,  # olivine_fo10
        # 4. Silica
        (192,192,255):  4,  # silica
        # 5: Ilmenite
        (0,255,0):      5,  # ilmenite (titanium ore)
        # 6: Chromite
        (0,128,255):    6,  # chromite (spinel group)
        # 7: Mafic Glass
        (170,232,238):  7,  # mafic_glass
        # 8: Ferropigeonite
        (170,15,210):   8,  # ferropigeonite
        # 9: Unclassified
        (255,255,255):  0   # background
        }

    contrast_index_pyx = {
        # classify only minerals within Pyroxene group (exc Ferropigeonite)
        (223,155,180):  255,
        (245,120,230):  255,
        (235,15,210):   255,
        (245,35,149):   255
        }
    
    contrast_index_plg = {
        # classify only minerals within Plagioclase Feldspar group
        (128,128,0):    255,
        (175,175,0):    255,
        (244,244,0):    255
        }
    
    contrast_index_olv = {
        # classify only minerals within Olivine group
        (0,190,190):    255,
        (0,170,170):    255,
        (0,150,150):    255,
        (0,130,130):    255,
        (0,110,110):    255,
        (0,90,90):      255,
        (0,70,70):      255,
        (0,50,50):      255,
        }

    # define data folders per sample
    folder_1 = r'data\15125,6'
    folder_2 = r'data\15475,15'
    folder_3 = r'data\15555,209'
    folder_4 = r'data\15597,18'

    # create labels
    in_source = '*.png'
    clf, clf_contrast = create_labels(folder_1, in_source, label_index, contrast_index_pyx, remove_noise=True)
    
    # denoise classifier
    in_clf = '*.png'
    clf_f, clg_g = clf_denoise(folder_1, in_clf, label_index)
       
    # align images and create feature creation inputs
    in_img = '*.png'
    in_clf_contrast = '*.png'
    in_clf = '*.png'
    img_f, clf_f, clf_brg = create_aligned_inputs(folder_1, in_img, in_clf_contrast, in_clf, one_pass_only=False, save_all=True)
    
    # align ppl/xpl images
    ppl = '*.png'
    xpl = '*.png'
    img_align = align_ppl_xpl(folder_1, ppl, xpl)

