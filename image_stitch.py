'''
David Bloomer (13161521)
Birkbeck University, MSc Data Science 18-20 PT

References:
https://github.com/opencv/opencv/blob/master/samples/python/stitching.py
https://raw.githubusercontent.com/opencv/opencv/master/samples/python/stitching_detailed.py
'''

# opencv implementation of Brown and Lowe (2007)

import cv2 as cv
import glob

# variables
folder = r'data\15125,6'
ext = 'tif'
outfile = '*.png'

# code
imagePaths = glob.glob(f'{folder}\*.{ext}')

images = []
for imagePath in imagePaths:
	image = cv.imread(imagePath, cv.IMREAD_UNCHANGED)
	images.append(image)

stitcher = cv.Stitcher_create(mode=cv.Stitcher_SCANS)
status, stitched = stitcher.stitch(images)

if status == 0:
	cv.imwrite(f'{folder}\{outfile}', stitched)
else:
	lookup = {1:'Need more images',
              2:'Homography estimation failure',
              3:'Camera parameters adjust failure'}
	print(f'Stitch failed: {lookup[status]}')