# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--text_file", required=True,
	help="path to input image")
ap.add_argument("-r", "--race", required=True,
	help="path to input image")
args = ap.parse_args()


# download pre-trained model file (one-time download)
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())
f = open(args.text_file+".txt","w+")
images_root = os.path.join('/cmlscratch','dtinubu','datasets','RFW','Balancedface','race_per_7000', args.race)
names = os.listdir(images_root)
for klass, name in enumerate(names):
		  image_path = os.path.join(images_root, name)
		  images_of_person = os.listdir(os.path.join(images_root, name))
		  for  name in zip(images_of_person):
		    path = (image_path + name)
		    # read input image
		    image = cv2.imread(image_path)
	            if image is None:
		    	print("Could not read input image")
		    	exit()
			
		    # load pre-trained model
		    model = load_model(model_path)

			# detect faces in the image
		    face, confidence = cv.detect_face(image)
		    classes = ['man','woman']

	            # loop through detected faces
		    for idx, f in enumerate(face):

	            # get corner points of face rectangle       
		    (startX, startY) = f[0], f[1]
		    (endX, endY) = f[2], f[3]

			    # draw rectangle over face
			     cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

			    # crop the detected face region
			     face_crop = np.copy(image[startY:endY,startX:endX])

			    # preprocessing for gender detection model
			     face_crop = cv2.resize(face_crop, (96,96))
			     face_crop = face_crop.astype("float") / 255.0
			     face_crop = img_to_array(face_crop)
			     face_crop = np.expand_dims(face_crop, axis=0)

			    # apply gender detection on face
			     conf = model.predict(face_crop)[0]
			     print(conf)
			     print(classes)

			    # get label with max accuracy
			     idx = np.argmax(conf)
			     label = classes[idx]

			     label = "{}: {:.2f}%".format(label, conf[idx] * 100)

			     Y = startY - 10 if startY - 10 > 10 else startY + 10

			     # write label and confidence above face rectangle
			     f.write(label + "," + name + "," + image_path )


# press any key to close window
cv2.waitKey()
# save output
cv2.imwrite("gender_detection.jpg", image)


# release resources
cv2.destroyAllWindows()
