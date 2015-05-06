__author__ = 'Brian Diaz'
import cv2
from face import Face

cascade_path = "haarcascades/xml/haarcascade_frontalface_alt.xml"

class FaceDetector:

	def __init__(self, image, scale_factor, minimum_neighbors, minimum_size, flags):
		self.face_cascade = cv2.CascadeClassifier(cascade_path)
		self.image = image
		self.scale_factor = scale_factor
		self.minimum_neighbors = minimum_neighbors
		self.minimum_size = minimum_size
		self.flags = flags

	def __detect_faces(self):
		faces = self.face_cascade.detectMultiScale(
		    self.image,
		    scaleFactor=self.scale_factor,
		    minNeighbors=self.minimum_neighbors,
		    minSize=self.minimum_size,
		    flags = self.flags
		)
		return faces

	def detect_faces(self):
		faces = []
		rgb = (0, 255, 0)
		for (x, y, w, h) in self.__detect_faces():
			face = Face(self.image, x, y, w, h, rgb)
			face.draw_rectangle(2)
			face.show
			faces.append(face)
		return faces
