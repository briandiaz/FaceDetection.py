import cv2
from image import Image 
from detector import Detector
from classifier import Classifier
import time

video_capture = cv2.VideoCapture(0) # Create instance of videoCapture on Default WebCamera

haarcascade_classifier = Classifier("alt", "eye_glasses", "smile") # Create instance of Classifier with haarcascade classifier xml preferred.

video_capture.open(0) # Open Default WebCamera
time.sleep(5) # Wait Camera to respond

while True: 
	is_read, image = video_capture.read()
	if is_read:
		detector = Detector(image, 1.1, 5, (30, 30), cv2.DIST_L2, haarcascade_classifier)
		detector.detect()
		cv2.imshow('video', image)
	else:
		cv2.waitKey(1000)

	key = cv2.waitKey(30) & 0xff
	if key == 27: # If Escape key is pressed video-capture ends.
		break

video_capture.release()
cv2.destroyAllWindows()
