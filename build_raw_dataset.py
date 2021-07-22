
# Creating database
# It captures images and stores them in datasets 
# folder under the folder name of sub_data
import cv2, sys, numpy, os
from imutils.video import VideoStream
import imutils
import time

import argparse

# All the faces data will be
#  present this folder


def main(args):
    datasets = 'Dataset/FaceData/raw'  
    haar_file = 'src/Models/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    
    # These are sub data sets of folder, 
    # for my faces I've used my name you can 
    # change the label here
    sub_data = args.username    
    print (sub_data)
    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    
    cap  = VideoStream(src=0).start()


    # The program loops until it has 30 images of the face.
    count = 0
    while count < 100: 
        frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        cv2.imshow(sub_data, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.imwrite('% s/% s.png' % (path, count), frame)
            count += 1
            print (count)
            time.sleep(0.2) 

        
        key = cv2.waitKey(10)
        if key == 27:
            break
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('username', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
