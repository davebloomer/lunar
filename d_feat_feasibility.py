'''
David Bloomer (13161521)
Birkbeck University, MSc Data Science 18-20 PT
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.lib import stride_tricks
from skimage.util import img_as_ubyte, img_as_float
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.filters import scharr, prewitt, roberts, sobel, gabor, frangi, hessian, sato, meijering
from skimage.feature import canny, hessian_matrix
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from scipy.stats import linregress

from tqdm import tqdm

from b_feat_creation import img_load, clf_load, preproc_gaussian_filt, eq_adaptivehist

# Feasibility Analysis

def img_load_min(img, normalise=False, blur=True):
    '''
    Simplified image (img) load function, option to apply CLAHE image equalisation
    and apply Gaussian filter.
    ----------------------
    Parameters
    ----------------------
    img: str, file location
    normalise: Boolean (Default=False)
    normalise: Boolean (Default=False)
    ----------------------
    Returns
    ----------------------
    out: (M,N,3) ndarray
    '''
    img = img_load(img, crop=False)
    if normalise:
        img = eq_adaptivehist(img)
    if blur:
        img = preproc_gaussian_filt(img)
    return img

def img_border_crop(img, samp_dim):
    '''
    Border removed from image prior to search for idx positions, as some features
    require spatial analysis performed using window of dimension samp_dim.
    ----------------------
    Parameters
    ----------------------
    img: (M,N,3) ndarray
    samp_dim: int
    ----------------------
    Returns
    ----------------------
    out: (M,N,3) ndarray
    '''
    assert samp_dim % 2 == 1, 'samp_dim must not be even.'
    samp_border = int((samp_dim - 1)/ 2)
    return img[samp_border:-samp_border, samp_border:-samp_border]   

def build_index(clf, samp_n, samp_dim, seed=0):
    '''
    For input classification (clf), find labels present and return index positions
    for (samp_n) samples.
    ----------------------
    Parameters
    ----------------------
    clf: (M,N) ndarray
    samp_n: int
    samp_dim: int
    seed: int (Default=0)
    ----------------------
    Returns
    ----------------------
    class_list: list
    class_idx: list of lists
    '''
    # crop image to avoid samples excluded by samp_size
    clf = img_border_crop(clf, samp_dim)
    clf = clf.reshape((clf.shape[0]*clf.shape[1], 1))
    
    # find present classes and counts
    classes = np.unique(clf, return_counts=True)
    
    idx = np.indices((clf.shape))
    class_list, class_idx = [], []
    # exclude background/unclassified labels (:-1)
    for i, c in zip(classes[0][1:-1],classes[1][1:-1]):  
        # warn if sample size below samp_n for class
        if c < samp_n:
            print(f'Class {i} below minimum required sample count, excluded from analysis.')
        # else build index list per class
        else:
            e = idx[0][(clf == i)]
            np.random.seed(seed)
            np.random.shuffle(e)
            class_list.append(i)
            class_idx.append(e[:samp_n])
    
    return class_list, class_idx

def build_image_view(img, samp_dim):
    '''
    Utility function to reshape image array.
    '''
    shape = (img.shape[0]-samp_dim+1, img.shape[1]-samp_dim+1, samp_dim, samp_dim)
    strides = 2*img.strides
    view = stride_tricks.as_strided(img, shape=shape, strides=strides)
    return view.reshape(-1, samp_dim, samp_dim)

def return_feature_list(img, class_list, class_idx, feat):
    '''
    For samples within each class, return feature values at indexed pixel positions.
    ----------------------
    Parameters
    ----------------------
    img: (M,N,3) ndarray
    class_list: list
    class_idx: list of lists
    feat: func
    ----------------------
    Returns
    ----------------------
    out: list of lists
    '''
    feature_list = []
    
    # if colour, extract values using class_idx
    non_spatial_feats = [feat_greyscale, feat_rgb_r, feat_rgb_g, feat_rgb_b, feat_hsv_h,
                         feat_hsv_s, feat_hsv_v, feat_lab_l, feat_lab_a, feat_lab_b]
    if feat in non_spatial_feats:
        img = feat(img)
        img = img_border_crop(img, samp_dim)
        img = img.reshape((img.shape[0]*img.shape[1], 1))
        
        for c in range(len(class_list)):
            feature_list.append(list(img[class_idx[c]].flatten()))
            
    # else build image view (increase complexity)
    else:
        img = util_greyscale(img)
        view = build_image_view(img, samp_dim)
        
        for c in range(len(class_list)):
            class_feat = []
            for v in view[class_idx[c]]:
                class_feat.append(feat(v))
            feature_list.append(class_feat)
    
    return feature_list

def wasserstein(x1, x2, p1=1, p2=1):
    '''
    Wasserstein distance between two distributions.
    '''
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(x1, x2)

def feat_wasserstein_min(feature_list):
    '''
    For input feature_list, scale data for each class and simplify distribution
    to remove impact of sample bias on frequency. Return for each class label,
    the minimum Wasserstain to all other class label distributions.
    ----------------------
    Parameters
    ----------------------
    feature_list: list of lists
    ----------------------
    Returns
    ----------------------
    out: list
    '''  
    # standard scaler
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(np.array([x for y in feature_list for x in y]).reshape(-1,1))
    feature_list_scale = [sc.transform(np.array(i).reshape(-1,1)).squeeze().tolist() for i in feature_list]
    
    # simplified distribution
    med_std = [(np.median(i), np.std(i)) for i in feature_list_scale]
    feature_list_simp = [[med-std, med+std] for med, std in med_std]
    
    # Wasserstein
    feat_len = len(feature_list_simp)
    score_list = [999]*feat_len  
    for i in range(feat_len-1):
        for j in range(i+1, feat_len):
            #dist = [x for y in feature_list_scale[:i]+feature_list_scale[i+1:] for x in y]
            score = wasserstein(feature_list_simp[i], feature_list_simp[j])
            if score < score_list[i]:
                score_list[i] = score
            if score < score_list[j]:
                score_list[j] = score
    
    return score_list

def feat_colormap():
    '''
    Colormap defining labels and RGB colour for each class.
    '''
    return {0: ('background', (1.000, 1.000, 1.000)),
            1: ('pyroxene', (0.902, 0.471, 0.961)),
            2: ('plagioclase', (0.000, 0.502, 0.502)),
            3: ('olivine', (0.850, 0.850, 0.000)),
            4: ('silica', (1.000, 0.753, 0.753)),
            5: ('opaques', (0.000, 1.000, 0.000)),
            6: ('matrix', (0.400, 0.400, 0.400)),
            7: ('unknown', (0.000, 0.000, 0.000))}
    
def feat_importance(ppl, class_list, class_idx, features, xpl='none', plot=True):
    '''
    For each class, using all features within features list, calculate minimum
    Wasserstein distance metric for ppl and/or xpl inputs.
    Optional visualisation of class performance through bar chart.
    ----------------------
    Parameters
    ----------------------
    ppl: (M,N,3) ndarray
    class_list: list
    class_idx: list of lists
    features: list
    xpl: str, (M,N,3) ndarray
    plot: Boolean (Default=True)
    ----------------------
    Returns
    ----------------------
    out: pd.DataFrame
        Tabular feasibility analysis for each class.
    '''  
    colormap = feat_colormap()
    class_labels = [colormap[i][0] for i in class_list]
    dataframe = pd.DataFrame(columns=['feature']+class_labels)
    for i, (label, f) in tqdm(enumerate(features), desc='Creating ppl features'):
        feature_list = return_feature_list(ppl, class_list, class_idx, f)
        imp = feat_wasserstein_min(feature_list)
        dataframe.loc[i] = ['[p] '+label] + imp
    
    if xpl != 'none':
        for i, (label, f) in tqdm(enumerate(features), desc='Creating xpl features'):
            feature_list = return_feature_list(xpl, class_list, class_idx, f)
            imp = feat_wasserstein_min(feature_list)
            dataframe.loc[i+len(features)] = ['[x] '+label] + imp
    
    if plot:
        plot_feat_importance(dataframe, class_list)
    
    return dataframe

# Feature Engineering
# wrapper functions for scikit-image functionality

def util_greyscale(img):
    return rgb2gray(img)

def util_return_center(img):
    center = int(((img.shape[0]*img.shape[1])+1)/2)
    return img.item(center)

def feat_greyscale(img):
    return rgb2gray(img)

def feat_rgb_r(img):
    img = img_as_float(img)
    return img[:,:,0]

def feat_rgb_g(img):
    img = img_as_float(img)
    return img[:,:,1]

def feat_rgb_b(img):
    img = img_as_float(img)
    return img[:,:,2]

def feat_hsv_h(img):
    return rgb2hsv(img)[:,:,0]

def feat_hsv_s(img):
    return rgb2hsv(img)[:,:,1]

def feat_hsv_v(img):
    return rgb2hsv(img)[:,:,2]

def feat_lab_l(img):
    img = (rgb2lab(img) + [0, 128, 128]) / [100, 255, 255]
    return img[:,:,0]

def feat_lab_a(img):
    img = (rgb2lab(img) + [0, 128, 128]) / [100, 255, 255]
    return img[:,:,1]

def feat_lab_b(img):
    img = (rgb2lab(img) + [0, 128, 128]) / [100, 255, 255]
    return img[:,:,2]

def feat_canny(img, sigma=1):
    img = canny(util_greyscale(img), sigma=sigma)
    return util_return_center(img)

def feat_sobel(img):
    img = sobel(util_greyscale(img))
    return util_return_center(img)

def feat_prewitt(img):
    img = prewitt(util_greyscale(img))
    return util_return_center(img)

def feat_scharr(img):
    img = scharr(util_greyscale(img))
    return util_return_center(img)

def feat_roberts(img):
    img = roberts(util_greyscale(img))
    return util_return_center(img)

def feat_gabor(img, ang_inc=4, frequencies=[0.1,0.2,0.3,0.4]):
    img = util_greyscale(img)
    h, w = img.shape
    output_list = []
    for i in range(ang_inc):
        theta = i / 4. * np.pi  # eqv to (0,135,45) deg
        for freq in frequencies: 
            filt_real, filt_imag = gabor(img, theta=theta, frequency=freq)
            output_list.append(util_return_center((filt_real**2+filt_imag**2)**0.5))
    return output_list

def feat_gabor_freq_grad(img, ang_deg=0, frequencies=[0.1,0.2,0.3,0.4]):
    rad = ang_deg * (np.pi/180)
    value_list = []
    for freq in frequencies: 
        filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
        value_list.append(util_return_center((filt_real**2+filt_imag**2)**0.5))
    lin = linregress(range(len(frequencies)), value_list)
    return lin[0]

def feat_gabor_ang_grad(img, ang_inc=4, freq=0.3):
    img = util_greyscale(img)
    value_list = []
    for i in range(ang_inc):
        theta = i / 4. * np.pi  # eqv to (0,135,45) deg
        filt_real, filt_imag = gabor(img, theta=theta, frequency=freq)
        value_list.append(util_return_center((filt_real**2+filt_imag**2)**0.5))
    lin = linregress(value_list, range(ang_inc))
    return lin[0]

def feat_gabor_deg0_freq1(img, ang_deg=0, freq=0.1):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg0_freq2(img, ang_deg=0, freq=0.2):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg0_freq3(img, ang_deg=0, freq=0.3):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg0_freq4(img, ang_deg=0, freq=0.4):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg45_freq1(img, ang_deg=45, freq=0.1):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg45_freq2(img, ang_deg=45, freq=0.2):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg45_freq3(img, ang_deg=45, freq=0.3):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg45_freq4(img, ang_deg=45, freq=0.4):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg90_freq1(img, ang_deg=90, freq=0.1):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg90_freq2(img, ang_deg=90, freq=0.2):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg90_freq3(img, ang_deg=90, freq=0.3):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg90_freq4(img, ang_deg=90, freq=0.4):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg135_freq1(img, ang_deg=135, freq=0.1):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg135_freq2(img, ang_deg=135, freq=0.2):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg135_freq3(img, ang_deg=135, freq=0.3):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_gabor_deg135_freq4(img, ang_deg=135, freq=0.4):
    img = util_greyscale(img)
    rad = ang_deg * (np.pi/180)
    filt_real, filt_imag = gabor(img, theta=rad, frequency=freq)
    return util_return_center((filt_real**2+filt_imag**2)**0.5)

def feat_hessian(img):
    img = hessian(util_greyscale(img))
    return util_return_center(img)

def feat_hessian_rr(img):
    rr, rc, cc = hessian_matrix(util_greyscale(img), sigma=1, order='rc', mode='reflect')
    return rr

def feat_hessian_rc(img):
    rr, rc, cc = hessian_matrix(util_greyscale(img), sigma=1, order='rc', mode='reflect')
    return rc

def feat_hessian_cc(img):
    rr, rc, cc = hessian_matrix(util_greyscale(img), sigma=1, order='rc', mode='reflect')
    return cc

def feat_frangi(img):
    img = frangi(util_greyscale(img))
    return util_return_center(img)

def feat_sato(img):
    img = sato(util_greyscale(img))
    return util_return_center(img)

def feat_meijering(img):
    img = meijering(util_greyscale(img))
    return util_return_center(img)

def feat_haralick_contrast(img):
    img = img_as_ubyte(img)
    glcm = greycomatrix(img, [1], np.arange(4)/4*np.pi, levels=256, symmetric=True, normed=True)
    return greycoprops(glcm, 'contrast').mean(axis=1)[0]

def feat_haralick_dissimilarity(img):
    img = img_as_ubyte(img)
    glcm = greycomatrix(img, [1], np.arange(4)/4*np.pi, levels=256, symmetric=True, normed=True)
    return greycoprops(glcm, 'dissimilarity').mean(axis=1)[0]

def feat_haralick_homogeneity(img):
    img = img_as_ubyte(img)
    glcm = greycomatrix(img, [1], np.arange(4)/4*np.pi, levels=256, symmetric=True, normed=True)
    return greycoprops(glcm, 'homogeneity').mean(axis=1)[0]

def feat_haralick_energy(img):
    img = img_as_ubyte(img)
    glcm = greycomatrix(img, [1], np.arange(4)/4*np.pi, levels=256, symmetric=True, normed=True)
    return greycoprops(glcm, 'energy').mean(axis=1)[0]

def feat_haralick_correlation(img):
    img = img_as_ubyte(img)
    glcm = greycomatrix(img, [1], np.arange(4)/4*np.pi, levels=256, symmetric=True, normed=True)
    return greycoprops(glcm, 'correlation').mean(axis=1)[0]

def feat_haralick_ASM(img):
    img = img_as_ubyte(img)
    glcm = greycomatrix(img, [1], np.arange(4)/4*np.pi, levels=256, symmetric=True, normed=True)
    return greycoprops(glcm, 'ASM').mean(axis=1)[0]

def feat_lbp2(img, r=2):
    lbp = local_binary_pattern(util_greyscale(img), r*8, r)
    return util_return_center(lbp)

def feat_lbp4(img, r=4):
    lbp = local_binary_pattern(util_greyscale(img), r*8, r)
    return util_return_center(lbp)

def feat_lbp2_uniform(img, r=2):
    lbp = local_binary_pattern(util_greyscale(img), r*8, r, method='uniform')
    return util_return_center(lbp)

def feat_lbp4_uniform(img, r=4): # r <= 4 with sample_dim = 11
    lbp = local_binary_pattern(util_greyscale(img), r*8, r, method='uniform')
    return util_return_center(lbp)

# Reporting

def plot_univariate_density(class_list, feature_list):
    '''
    For reporting: Visualisation of probability density functions per class.
    '''
    colormap = feat_colormap()
    for i in range(len(class_list)):
        name, rgb = colormap[class_list[i]]
        sns.distplot(feature_list[i], hist=True, kde=True,
                     kde_kws={'linewidth': 2},
                     label=name,
                     color=rgb)
    
    plt.legend(title='Class')
    plt.show()

def plot_univariate_box(class_list, feature_list):
    '''
    For reporting: Visualisation of box plot per class.
    '''
    colormap = feat_colormap()
    labels = [colormap[i][0] for i in class_list]
    colors = [colormap[i][1] for i in class_list]
    
    bp = plt.boxplot(feature_list,
                     labels=labels,
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)   
    
    plt.show()
    
def plot_gabor(img, clf):
    '''
    For reporting: Visualisation of average Gabor values for range of angles
    and frequencies per class.
    '''
    img = util_greyscale(img)
    samp_dim = 11
    view = build_image_view(img, samp_dim)
    class_list, class_idx = build_index(clf, 30, samp_dim, seed=0)

    freqs = ['0.1','0.2','0.3','0.4']
    angles = ['0','45','90','135']
    ind = np.arange(len(freqs))
    width = 0.2
    colormap = feat_colormap()
    
    for i, c in enumerate(range(len(class_list))):
        # prepare data
        class_feat = np.empty((0,len(angles)*(len(freqs))))
        for v in view[class_idx[c]]:
            class_feat = np.vstack((class_feat, feat_gabor(v)))
        class_feat = np.mean(class_feat, axis=0)
        
        # plot
        ax = plt.subplot(1, len(class_list), c+1)
        name, rgb = colormap[class_list[c]]
        ax.set_ylim([0, 0.03])
        if i != 0:
            ax.set_yticklabels([])
        ax.set_xticklabels(['']+freqs)
        ax.set(xlabel=name)
        for a in range(len(angles)):
            plt.bar(ind+(a*width), class_feat[a*4:(a+1)*4], width, label=angles[a])
    
    plt.legend(title='Angle (deg)')
    plt.show()

def plot_feat_importance(feasibility, class_list):
    '''
    For reporting: Visualisation of feasiblity performance through bar chart.
    '''
    colormap = feat_colormap()
    labels = [colormap[i][0] for i in class_list]
    colors = [colormap[i][1] for i in class_list]
    
    ax = plt.subplot(1, 1, 1)
    plt.bar(labels, feasibility.max(axis=0)[1:], color=colors)
    ax.set_ylabel('Wasserstein metric')
    plt.show()

#%%############################################################################

def main():
    '''
    Pseudo-code to demonstrate applied workflow.
    '''
    # load composite data
    ppl = img_load_min(r'data\composite\comp_ppl.png', normalise=False, blur=True)
    xpl = img_load_min(r'data\composite\comp_xpl.png', normalise=False, blur=True)
    clf = clf_load(r'data\composite\comp_clf.png', crop=False)
    #plt.imshow(ppl)
    #plt.imshow(clf2rgb(clf))
    
    # example of possible features
    features = [('greyscale', feat_greyscale),
                ('r (rgb)', feat_rgb_r),('g (rgb)', feat_rgb_g),('b (rgb)', feat_rgb_b),
                ('h (hsv)', feat_hsv_h),('s (hsv)', feat_hsv_s),('v (hsv)', feat_hsv_v),
                ('l (cielab)', feat_lab_l),('a (cielab)', feat_lab_a),('b (cielab)', feat_lab_b),
                ('sobel (edge)', feat_sobel),('canny (edge)', feat_canny),
                ('0 deg 0.1 freq (gabor)', feat_gabor_deg0_freq1),
                ('0 deg 0.2 freq (gabor)', feat_gabor_deg0_freq2),
                ('0 deg 0.3 freq (gabor)', feat_gabor_deg0_freq3),
                ('0 deg 0.4 freq (gabor)', feat_gabor_deg0_freq4),
                ('45 deg 0.1 freq (gabor)', feat_gabor_deg45_freq1),
                ('45 deg 0.2 freq (gabor)', feat_gabor_deg45_freq2),
                ('45 deg 0.3 freq (gabor)', feat_gabor_deg45_freq3),
                ('45 deg 0.4 freq (gabor)', feat_gabor_deg45_freq4),            
                ('90 deg 0.1 freq (gabor)', feat_gabor_deg90_freq1),
                ('90 deg 0.2 freq (gabor)', feat_gabor_deg90_freq2),
                ('90 deg 0.3 freq (gabor)', feat_gabor_deg90_freq3),
                ('90 deg 0.4 freq (gabor)', feat_gabor_deg90_freq4),
                ('135 deg 0.1 freq (gabor)', feat_gabor_deg135_freq1),
                ('135 deg 0.2 freq (gabor)', feat_gabor_deg135_freq2),
                ('135 deg 0.3 freq (gabor)', feat_gabor_deg135_freq3),
                ('135 deg 0.4 freq (gabor)', feat_gabor_deg135_freq4),
                ('freq gradient (gabor)', feat_gabor_freq_grad),
                ('hessian (ridge)', feat_hessian),
                ('frangi (ridge)', feat_frangi),
                ('sato (ridge)', feat_sato),
                ('meijering (ridge)', feat_meijering),
                ('n2 (lbp)', feat_lbp2),
                ('n4 (lbp)', feat_lbp4),
                ('n2 uniform (lbp)', feat_lbp2_uniform),
                ('n4 uniform (lbp)', feat_lbp4_uniform),
                ('contrast (haralick)*', feat_haralick_contrast),
                ('dissimilarity (haralick)*', feat_haralick_dissimilarity),
                ('homogeneity (haralick)*', feat_haralick_homogeneity),
                ('energy (haralick)*', feat_haralick_energy),
                ('correlation (haralick)*', feat_haralick_correlation),
                ('ASM (haralick)*', feat_haralick_ASM)                
                ]

    # define sampling strategy
    seed = 0
    samp_n = 500
    samp_dim = 11
    
    # find classes present and associated idxs
    class_list, class_idx = build_index(clf, samp_n, samp_dim, seed=seed)

    # create single feature and plot
    feature_list = return_feature_list(ppl, class_list, class_idx, feat_greyscale)
    feature_list = return_feature_list(xpl, class_list, class_idx, feat_rgb_b)
    
    plot_univariate_density(class_list, feature_list)
    plot_univariate_box(class_list, feature_list)
    plot_gabor(ppl, clf)

    # feasbility analysis
    feasibility = feat_importance(ppl, class_list, class_idx, features, xpl=xpl, plot=True)
    #plot_feat_importance(feasibility, class_list)

