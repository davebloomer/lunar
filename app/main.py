'''
David Bloomer (13161521)
Birkbeck University, MSc Data Science 18-20 PT
'''

import streamlit as st

import numpy as np
import pandas as pd

from joblib import load

from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv, rgb2lab, label2rgb
from skimage.filters import gaussian, sobel, gabor, frangi, sato, meijering
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops_table
from skimage.future.graph import rag_mean_color, cut_threshold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
    st.title('Lunar Rock Composition Analysis')
    
    # Load Classifier
    model = load(r'..\model\multi_min_final.clf')
    
    # SIDEBAR

    # Image Selector
    sample_images = {'15125, Split 6':'images/15125_6',
                     '15475, Split 15':'images/15475_15',
                     '15555, Split 209':'images/15555_209',
                     '15597, Split 18':'images/15597_18',
                     'Custom':''}
    sample = st.sidebar.selectbox('Select Images:', list(sample_images.keys()))
    
    if sample == 'Custom':
        in_ppl = st.sidebar.file_uploader('PPL', type='PNG')
        in_xpl = st.sidebar.file_uploader('XPL', type='PNG')
    else:
        filename = sample_images[sample]
        in_ppl = filename+'.png'
        in_xpl = filename+'a.png'

    # Once images are selected..
    if in_ppl is None or in_xpl is None:
        st.write('Please select input image(s).')
    else:
        
        # Select Optional Panels
        st.sidebar.markdown('Select interpretation(s):')
        view_bulk_comp = st.sidebar.checkbox('Bulk Composition')
        #view_rock_type = st.sidebar.checkbox('Rock Classification')
        view_texture = st.sidebar.checkbox('Texture')
        if view_texture:
            slic_n = st.sidebar.slider('Select n_segments value:', 10, 1000, value=400, step=10)
            slic_c = st.sidebar.slider('Select compactness value (rigidity):', 0.01, 100.0, value=40.0, step=1.0)
            slic_s = st.sidebar.slider('Select sigma value (edge smoothing):', 0.0, 3.0, value=1.0, step=0.2)
            rag_thresh = st.sidebar.slider('Select merging theshold:', 25, 50, value=40, step=5)
        #view_rock_phys = st.sidebar.checkbox('Elastic Response')
    
        # MAIN
        
        # Load Images and Perform Classification
        ppl, xpl = imread(in_ppl), imread(in_xpl)
        x, shape = create_features(ppl, xpl)
        clf = predict_clf(model, x, shape)
        clf_rgb = clf2rgb(clf)
        
        # Interpretation Overlay
        if view_texture:
            labels = slic(clf_rgb, n_segments=slic_n, compactness=slic_c, sigma=slic_s)
            g = rag_mean_color(np.uint(clf_rgb*255), labels)
            labels = cut_threshold(labels, g, rag_thresh)
            clf_plot = mark_boundaries(clf_rgb, labels)
        else:
            clf_plot = clf_rgb
    
        # Interpretation
        st.header('Interpretation')
        st.image([ppl, xpl, clf_plot],
                 caption=['PPL','XPL','Classification'],
                 width=200,
                 output_format='PNG')

        # Download
        #if st.button('Save'):
        #    st.markdown(f'<a href={clf_rgb}</a>', unsafe_allow_html=True)
    
        # Optional Panels
        # Bulk Composition
        if view_bulk_comp:
            st.header('Bulk Composition')
            st.write(prop_bulk_comp(clf))
    
        # Texture
        if view_texture:
            st.header('Texture')
            avg_orientation, avg_roundness = prop_texture(labels)
            st.write(f'Average Orientation: {avg_orientation} deg')
            st.write(f'Average Roundness: {avg_roundness}')

# Processing Functions

def preproc_gaussian_filt(img, sigma=3):
    return gaussian(img, sigma=sigma, multichannel=True)

def feat_greyscale(img):
    return rgb2gray(img)

def feat_la(img):
    return rgb2lab(img)[:,:,:2]

def feat_sobel(img):
    img = feat_greyscale(img)
    return sobel(img)

def feat_gabor(img, ang_inc=4, freqs=[0.1,0.2,0.3,0.4]):
    img = feat_greyscale(img)
    h, w = img.shape
    feat_map = np.zeros(shape=(h,w,ang_inc*len(freqs)))
    for i in range(ang_inc):
        theta = i / 4. * np.pi  # eqv to (0,35,45) deg
        for j, freq in enumerate(freqs): 
            d = (i*len(freqs))+j  # dimension of feat_map to calculate
            filt_real, filt_imag = gabor(img, theta=theta, frequency=freq)
            feat_map[:,:,d] = (filt_real**2+filt_imag**2)**0.5
    return feat_map

def feat_frangi(img):
    img = feat_greyscale(img)
    return frangi(img)

def feat_sato(img):
    img = feat_greyscale(img)
    return sato(img)  # mode='reflect'

def feat_meijering(img):
    img = feat_greyscale(img)
    return meijering(img)

def feat_lbp_uniform(img, r):
    img = feat_greyscale(img)
    return local_binary_pattern(img, r*8, r, method='uniform')

def array_flatten(array):
    h, w, d = array.shape
    return array.reshape((h*w, d))

def array_restore(array, shape):
    h, w = shape
    return array.reshape((h, w))

def create_features(ppl, xpl):
    h, w, _ = ppl.shape
    ppl, xpl = preproc_gaussian_filt(ppl), preproc_gaussian_filt(xpl)
    x = np.dstack([ppl, xpl,                      # rgb (colour space)
                   rgb2hsv(ppl),                  # hsv (colour space)
                   rgb2hsv(xpl),
                   feat_la(ppl),                  # cielab (colour space) 
                   feat_la(xpl),
                   feat_sobel(ppl),               # sobel (edge)
                   feat_sobel(xpl),
                   feat_gabor(ppl, freqs=[0.1]),  # gabor (filter bank)
                   feat_gabor(xpl, ang_inc=1, freqs=[0.1]),
                   feat_frangi(ppl),              # frangi (ridge)
                   feat_sato(ppl),                # sato (ridge)
                   feat_sato(xpl),
                   feat_meijering(ppl),           # meijering (ridge)
                   feat_meijering(xpl),
                   feat_lbp_uniform(ppl, r=4),    # uniform lbp (neighbourhood)
                   feat_lbp_uniform(ppl, r=8)
                   ])
    
    return array_flatten(x), (h,w)

def predict_clf(model, x, shape):
    yhat = model.predict(x)
    return array_restore(yhat, shape)

def clf2rgb(img):
    colormap = np.array([[1.000, 1.000, 1.000],   # 0: background
                         [0.902, 0.471, 0.961],   # 1: pyroxene
                         [0.000, 0.502, 0.502],   # 2: plagioclase
                         [0.850, 0.850, 0.000],   # 3: olivine
                         [1.000, 0.753, 0.753],   # 4: silica
                         [0.000, 1.000, 0.000],   # 5: opaques
                         [0.400, 0.400, 0.400],   # 6: matrix
                         [0.000, 0.000, 0.000]])  # 7: unknown
    colors_present = np.unique(img).astype(int)
    colormap = colormap[colors_present]  # index colormap using available labels
    
    return label2rgb(img, colors=colormap)

def prop_bulk_comp(clf):
    labelmap = {0: 'Background',
                1: 'Pyroxene',
                2: 'Plagioclase',
                3: 'Olivine',
                4: 'Silica',
                5: 'Opaques',
                6: 'Matrix',
                7: 'Unclassified'}
    comp_properties = ['label', 'area']
    vol_fract = regionprops_table(np.uint8(clf), properties=comp_properties)
    pd_vol_fract = pd.DataFrame(vol_fract).sort_values(by='area', ascending=False)
    pd_vol_fract['label'] = pd_vol_fract['label'].map(labelmap)
    pd_vol_fract['Fraction'] = pd_vol_fract['area'] / pd_vol_fract['area'].sum()
    return pd_vol_fract.set_index('label').drop(columns=['area'])

def prop_texture(labels):
    texture_properties = ['centroid', 'orientation', 'major_axis_length', 'minor_axis_length']
    texture = regionprops_table(labels, properties=texture_properties)
    pd_texture = pd.DataFrame(texture)
    pd_texture['roundness'] = pd_texture['major_axis_length'] / pd_texture['minor_axis_length']
    avg_orientation = pd_texture['orientation'].mean()
    avg_roundness = pd_texture['roundness'].mean()
    return round(avg_orientation*(180/(avg_orientation*np.pi)),2), round(avg_roundness,2)

if __name__ == "__main__":
    main()

