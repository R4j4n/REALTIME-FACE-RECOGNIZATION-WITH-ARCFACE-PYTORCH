import os 
import cv2
import pickle
import numpy as np 
from libs.arcface.arcface import Arcface
from sklearn.preprocessing import Normalizer
l2_normalizer = Normalizer('l2')
 
model =  Arcface()
from PIL import Image
face_data = "FaceName\\"
encoding_dict = {}
for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data,face_names)
    
    embeddings = [] 
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir,image_name)
        
        img = cv2.imread(image_path)
        encode = model(img)
        encode = encode[0]
        embeddings.append(encode)

    
    if embeddings:
        encode = np.sum(embeddings,axis= 0 )
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        encoding_dict[face_names] = encode
        
path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file) 
    
    

