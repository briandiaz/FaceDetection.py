__author__ = 'Brian Diaz'
import cv2
from face import Face

xml_path = "haarcascades/xml/"

cascade_path = {
	"alt": xml_path + "haarcascade_frontalface_alt.xml",
	"alt2": xml_path + "haarcascade_frontalface_alt2.xml",
	"alt3": xml_path + "haarcascade_frontalface_alt_tree.xml",
	"default": xml_path + "haarcascade_frontalface_default.xml"
}

class FaceDetector:

	def __init__(self, image, cascade_trainer, scale_factor, minimum_neighbors, minimum_size, flags):
		self.cascade_trainer = cascade_path[cascade_trainer]
		self.face_cascade = cv2.CascadeClassifier(self.cascade_trainer)
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
