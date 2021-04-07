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
images_root = os.path.join('/cmlscratch','dtinubu','datasets','RFW','Balancedface','race_per_7000', args.race)
names = os.listdir(images_root)
for klass, name in enumerate(names):
		  image_path = os.path.join(images_root, name)
		  images_of_person = os.listdir(os.path.join(images_root, name))
		  for  i in range (len(images_of_person)):
                    path =  os.path.join(images_root, name, images_of_person[i])
		    # read input image
                    image = cv2.imread(path)
                    if image is None:
                      print("Could not read input image")
                      exit()
			faces, confidences = cv.detect_face(image) 
		        label, confidence = cv.detect_gender(face)
                        f = open(args.text_file+".txt","a")
                        f.write(label + "," + name + "," + path + "\n")
                        print(label + "," + name + "," + path + "\n")
                        f.close()


# press any key to close window
cv2.waitKey()
# save output
cv2.imwrite("gender_detection.jpg", image)


# release resources
cv2.destroyAllWindows()
