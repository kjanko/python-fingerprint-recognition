import cv2
import os
import sys
import numpy
import matplotlib.pyplot as plt

from skimage.feature import corner_harris, corner_subpix, corner_peaks, match_descriptors
from skimage.morphology import skeletonize
from skimage.util import invert

os.chdir("C:\Users\kjanko\Desktop\Fingerprint Recognition")

def get_descriptors(img):
	# Invert
	img = invert(img)
	# Threshold
	ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU);
	# Normalize to 0 and 1 range
	img[img == 255] = 1
	# Thinning
	skeleton = skeletonize(img)
	# Harris corners detection
	coords = corner_peaks(corner_harris(skeleton), min_distance=5)
	# Select corners
	coords_subpix = corner_subpix(skeleton, coords, window_size=13)
	# Define descriptor
	orb = cv2.ORB_create()
	keypoints = []
	for row in coords_subpix:
		keypoints.append(cv2.KeyPoint(x=row[0], y=row[1], _size=1))
	# Compute descriptors
	keypoints, des = orb.compute(img, keypoints)
	
	#Display keypoints
	"""fig, ax = plt.subplots()
	ax.imshow(skeleton, interpolation='nearest', cmap=plt.cm.gray)
	ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
	ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
	ax.axis((0, 350, 350, 0))
	plt.show()"""
	
	return (keypoints, des);


def main():
	image_name = sys.argv[1]
	img1 = cv2.imread("database/" + image_name, cv2.IMREAD_GRAYSCALE)
	kp1, des1 = get_descriptors(img1)
	
	image_name = sys.argv[2]
	img2 = cv2.imread("database/" + image_name, cv2.IMREAD_GRAYSCALE)
	kp2, des2 = get_descriptors(img2)
	# Hamming brute-force matching
	sk_matches = match_descriptors(des1, des2, metric='hamming')
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance, reverse=True)
	
	# Draw first matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, flags=2, outImg=img1)
	plt.imshow(img3)
	plt.show()
	
	score_threshold = 500;
	score = 0;
	
	for row in sk_matches:
		score += abs(row[0]-row[1])
	
	print(score)
	if score_threshold > score:
		print("Fingerprint matches!");
	else:
		print("Fingerprint does not match!")
 
if __name__ == "__main__":
	try:
		main()
	except:
		raise