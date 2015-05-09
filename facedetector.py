__author__ = 'Brian Diaz'
import cv2
from face import Face
from eye import Eye
import numpy as np


xml_path = "haarcascades/xml/"

cascade_path = {
	"alt": xml_path + "haarcascade_frontalface_alt.xml",
	"alt2": xml_path + "haarcascade_frontalface_alt2.xml",
	"alt3": xml_path + "haarcascade_frontalface_alt_tree.xml",
	"default": xml_path + "haarcascade_frontalface_default.xml",
	"eye": xml_path + "haarcascade_eye.xml",
	"eye_glasses": xml_path + "haarcascade_eye_tree_eyeglasses.xml",
}

class FaceDetector:

	def __init__(self, image, scale_factor, minimum_neighbors, minimum_size, flags, face_cascade_trainer, eye_cascade_trainer):
		self.face_cascade = cv2.CascadeClassifier(cascade_path[face_cascade_trainer])
		self.eye_cascade = cv2.CascadeClassifier(cascade_path[eye_cascade_trainer])
		self.image = image
		self.grayscale_image = self.__convert_image_to_gray(image)
		self.scale_factor = scale_factor
		self.minimum_neighbors = minimum_neighbors
		self.minimum_size = minimum_size
		self.flags = flags

	def __convert_image_to_gray(self, image):
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	def __detect_faces(self):
		faces = self.face_cascade.detectMultiScale(
		    self.grayscale_image,
		    scaleFactor=self.scale_factor,
		    minNeighbors=self.minimum_neighbors,
		    minSize=self.minimum_size,
		    flags = self.flags
		)
		return faces

	def __detect_eyes(self, image):
		eyes = self.eye_cascade.detectMultiScale(
		    self.__convert_image_to_gray(image)
		)
		return eyes

	def detect(self):
		faces = []
		eyes = []
		rgb_green = (0, 255, 0)
		rgb_blue = (0, 0, 255)
		for (x, y, w, h) in self.__detect_faces():
			face = Face(self.image, x, y, w, h, rgb_green)
			face.draw(2)
			faces.append(face)
			detected_eyes = self.__detect_eyes(face.data())
			if len(detected_eyes) > 0:
				for(eye_x, eye_y, eye_w, eye_h) in detected_eyes:
					eye = Eye(self.image, face.x + eye_x, face.y + eye_y, eye_w, eye_h, rgb_blue)
					eye.draw(2)
			        eyes.append(eye)
		return { "faces" : faces, "eyes" : eyes}
