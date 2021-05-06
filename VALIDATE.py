from libs.mtcnn.mtcnn import MTCNN
from libs.arcface.arcface import Arcface
import cv2
from libs.utils.align import *
from libs.utils.utils import *
import pickle
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
l2_normalizer = Normalizer('l2')
import numpy as np
import datetime


required_size = (112,112)
recognition_t=0.7


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def get_face(img, box):
    x1, y1, = box[0] , box[1]
    x2, y2 = box[2], box[3]
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face, (x1, y1), (x2, y2)

def detect(img ,detector,encoder,encoding_dict):

    landmarks, bounding_box = detector(img)
    bounding_boxes = np.asarray(bounding_box, dtype=int)

    for bounding_box in bounding_boxes:
        
        face , pt_1, pt_2 = get_face(img,bounding_box)
        # face = get_aligned_faces(img, bounding_box, landmarks)
        encode = encoder(face)[0]
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]

        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img 
    



if __name__ == "__main__":
    
    face_detector = MTCNN()
    face_encoder = Arcface()
    encodings_path = 'encodings\encodings.pkl'
    encoding_dict = load_pickle(encodings_path)
    cap = cv2.VideoCapture(0)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    
    while cap.isOpened():
        ret,frame = cap.read()
        
        # if not ret:
        #     print("CAM NOT OPEND") 
        #     break
        
        frame = detect(frame , face_detector , face_encoder , encoding_dict)
        total_frames = total_frames + 1
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow('camera', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
