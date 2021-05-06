import os
import cv2
from libs import FaceModel
from libs.utils.utils import show_bboxes
from libs.utils.align import warp_and_crop_face, get_reference_facial_points, get_aligned_faces
from libs.mtcnn.mtcnn import MTCNN



mtcnn = MTCNN()
cap = cv2.VideoCapture(0)



if __name__ == "__main__":
    
    who = input("Enter your Name: ")
    directory = "FaceName" + "\\" + str(who)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print("YOUR FACE IMAGES ALREADY EXISTS")
        exit(0)
        
    count = 0
    img_count = 0
    while True:
        count += 1 
        img_count += 1
        _,frame=cap.read()
        frame = cv2.flip(frame, flipCode = 1 )
        names = []
        names.append(who)
        
        landmarks, bboxs = mtcnn(frame)
        faces = get_aligned_faces(frame, bboxs, landmarks)
        frame = show_bboxes(frame , bboxs , landmarks , names)
        

        for face in faces:
             
            if count % 10 == 0: 

                cv2.imwrite( directory + "\\" + str(who) + "_" +str(count) + ".jpg", face)
        
        cv2.imshow("FRAME",frame)
        
        if img_count == 300:
            exit(0)
        key=cv2.waitKey(15)
        if key==ord('q'):
            break

    cv2.destroyAllWindows()