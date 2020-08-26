#!/usr/bin/env python3
"""
Lowpolify any image using Delaunay triangulation

Author: Ryan Larson, originally from @ghostwriternr: https://github.com/ghostwriternr/lowpolify
"""
from os.path import abspath, dirname, join, exists
from collections import defaultdict
from typing import Tuple, List
import cv2, dlib, numpy as np
from scipy.spatial import Delaunay
from skimage.draw import polygon as draw_polygon
from shapely.geometry import Polygon

# Path to predictor used in face detection model
DEFAULT_LANDMARKS_PATH = join(abspath(dirname(__file__)), "imgs", "shape_predictor_68_face_landmarks.dat")

DEFAULT_CANNY_A: int = 50
DEFAULT_CANNY_B: int = 55
DEFAULT_FRACTION_PERCENT: float = 0.15

DEFAULT_BG_BGR = [255, 255, 255]
DEFAULT_BG_PAD = 100


def create_triangles(
	image: np.ndarray,
	gray_image: np.ndarray,
	mask: np.ndarray,
	a: int = 50,
	b: int = 55,
	c: float = 0.1,
	show: bool = False):
	""" Returns triangulations
		Given a raw portrait image, and its computed BW & Masked versions,
		1. Search for faces, add landmarks
		2. Add bottom left/right points for shoulders
		3. Add points falling within mask
		4. Create Delauney Triangles
		5. Filter triangles whose centroid does not intersect mask
	"""
	height_image, width_image = image.shape[:2]
	points: np.ndarray

	gray_gpu = cv2.cuda_GpuMat()
	gray_gpu.upload(gray_image)

	# Using canny edge detection. Reference: http://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
	# First argument : Input image
	# Second argument: minVal (argument 'a')
	# Third argument : maxVal (argument 'b')
	canny = cv2.cuda.createCannyEdgeDetector(a, b)
	edges = canny.detect(gray_gpu)
	edges = edges.download()

	## CUDA
	# num_points = int(cv2.cuda.countNonZero(edges) * c)
	num_points = int(np.where(edges)[0].size * c)
	r, c = np.nonzero(edges)
	rnd  = np.zeros(r.shape) == 1
	rnd[:num_points]	= True
	np.random.shuffle(rnd)
	points 				= np.vstack([r[rnd], c[rnd]]).T
	# print(f"# canny_pts{num_points}")

	# # Using DLib to find facial landmarks
	detector  = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('/content/drive/My Drive/tmp/shape_predictor_68_face_landmarks.dat')
	dets 	  	= detector(image, 1)

	if (len(dets) > 0):
		shape 		= predictor(image, dets[0])
		points 		= np.vstack([points, [[shape.part(i).y, shape.part(i).x]
		                               for i in range(shape.num_parts)]])

	# # Filter to return only the points that fall within mask
	# # if (mask[tuple(pt)] > 0 if ((pt[1] < mask.shape[1]))  else False)])
	filter_points = np.array([pt for pt in points if (mask[tuple(pt)] > 0)])

	# # Create Delauney triangles
	delaunay         = Delaunay(filter_points, incremental=True)
	delaunay.close()
	triangles        = delaunay.points[delaunay.simplices]
	filter_triangles = np.array([tri for tri in triangles
			if (mask[int(Polygon(tri).centroid.x), int(Polygon(tri).centroid.y)] > 0)])

	return filter_triangles

def render_triangles(triangles: np.ndarray, image: np.ndarray, fill=[255, 255, 255]):
  height, width, channels = image.shape
  output = np.full((height, width, channels), fill_value=fill, dtype=image.dtype)

  for tri in triangles:
    rr, cc = draw_polygon(tri[:, 0], tri[:, 1], (height, width))
    color  = np.mean(image[rr, cc], axis=0)
    cv2.fillConvexPoly(output, tri[:, ::-1].astype('int32'), color)
    # output[rr, cc] = color
  return output

def add_border(
	image: np.ndarray,
	pad: int = DEFAULT_BG_PAD,
	color: np.ndarray = DEFAULT_BG_BGR) -> np.ndarray:
	height, width   = image.shape[:2]
	max_dim  				= max(height, width) + (2 * pad)
	pad_more 				= (max_dim - min(height, width)) // 2

	# Assign paddings
	landscape = width > height
	top       = pad_more if (landscape) else pad
	right     = pad if (landscape) else pad_more
	bottom    = pad_more if (landscape) else pad
	left      = pad if (landscape) else pad_more

	return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def mask_to_polygons(mask: np.ndarray, epsilon: float = 10., min_area: float =10.) -> List[Polygon]:
	"""Convert a mask ndarray (binarized image) to Multipolygons"""
	# first, rotate mask 180 because its flipped
	mask_copy = mask.copy()[::-1, :]

	# second, find contours with cv2: it's much faster than shapely
	contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

	if not contours:
			return MultiPolygon()

	# now messy stuff to associate parent and child contours
	cnt_children   = defaultdict(list)
	child_contours = set()
	assert hierarchy.shape[0] == 1

	# http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
	for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
			if parent_idx != -1:
					child_contours.add(idx)
					cnt_children[parent_idx].append(contours[idx])

	# create actual polygons filtering by area (removes artifacts)
	all_polygons = []
	for idx, cnt in enumerate(contours):
			if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
					assert cnt.shape[1] == 1
					poly = Polygon(shell=cnt[:, 0, :], holes=[c[:, 0, :] for c in cnt_children.get(idx, []) if cv2.contourArea(c) >= min_area])
					all_polygons.append(poly)

	return all_polygons

def helper(inImage, c=0.3, outImage=None, show=False):
	'''Helper function'''
	# Read
	highpoly_image = cv2.imread(inImage)

	# Call 'pre_process' function
	highpoly_image, noiseless_highpoly_image = pre_process(highpoly_image, 1200)

	# Thresholding
	# Use Otsu's method for calculating thresholds
	gray_image  = cv2.cvtColor(noiseless_highpoly_image, cv2.COLOR_BGR2GRAY)
	ycbcr_image = cv2.cvtColor(noiseless_highpoly_image, cv2.COLOR_RGB2YCrCb)

	for xdim in range(ycbcr_image.shape[0]):
		for ydim in range(ycbcr_image.shape[1]):
			ycbcr_image[xdim][ydim] = ycbcr_image[xdim][ydim][0]
	ycbcr_image = ycbcr_image[:, :, 0]

	clahe                  = cv2.createCLAHE()
	normalized_gray_image  = clahe.apply(gray_image)
	high_thresh, thresh_im = cv2.threshold(ycbcr_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	low_thresh         = 0.5 * high_thresh
	blurred_gray_image = cv2.GaussianBlur(gray_image, (0, 0), 3)
	sharp_gray_image   = cv2.addWeighted(gray_image, 2.5, blurred_gray_image, -1, 0)

	# Call 'get_triangulation' function: COMPUTES TRIANGLES
	tris = get_triangulation(highpoly_image, sharp_gray_image, low_thresh, high_thresh, c, show)

	# Call 'get_lowpoly' function: RENDERS TRIANGLES
	lowpoly_image = get_lowpoly(tris, highpoly_image)

	# Resize
	if np.max(highpoly_image.shape[:2]) < 750:
		scale         = 750 / float(np.max(highpoly_image.shape[:2]))
		lowpoly_image = cv2.resize(lowpoly_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

	if outImage is not None:
		cv2.imwrite(outImage, lowpoly_image)

	return lowpoly_image



class Polygonz:
	def __init__(self, landmark_path: str = DEFAULT_LANDMARKS_PATH):
		# if not (exists(landmark_path)):
		self.detector = dlib.get_frontal_face_detector()
		self.landmark = dlib.shape_predictor(landmark_path)

	def canny_edge_points(self,
		gray: np.ndarray,
		a: int = DEFAULT_CANNY_A,
		b: int = DEFAULT_CANNY_B,
		c: float = 0.1) -> np.ndarray:
		edges 		       = cv2.Canny(gray, a, b)
		r, c             = np.nonzero(edges)
		num_points       = int(np.where(edges)[0].size * c)
		rnd 						 = np.zeros(r.shape) == 1
		rnd[:num_points] = True
		np.random.shuffle(rnd)
		points = np.vstack([r[rnd], c[rnd]]).T
		return points
