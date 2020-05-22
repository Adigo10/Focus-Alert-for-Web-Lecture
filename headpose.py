# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2

#initializing dlib's shape_predictor_68_face_landmarks.dat and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# start the video stream thread
print("starting video stream thread...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)

def cal_angle(a,b,c):
    v1 = np.array(a) - np.array(b)
    v0 = np.array(b) - np.array(c)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

while True:
    _,frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape0 = predictor(gray, rect)
        shape0 = np.array(face_utils.shape_to_np(shape0))

        #2D image points. If you change the image, you need to change vector
        image_points = np.array([
                                    (shape0[33, :]),     # Nose tip
                                    (shape0[8,  :]),     # Chin
                                    (shape0[36, :]),     # Left eye left corner
                                    (shape0[45, :]),     # Right eye right corne
                                    (shape0[48, :]),     # Left Mouth corner
                                    (shape0[54, :])      # Right mouth corner
                                ], dtype="double")

        # 3D model points.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                ])

        # Camera internals
        size = frame.shape
        focal_length = size[1]
        center = (size[1]//2, size[0]//2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
        #print ("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #print ("Rotation Vector:\n {0}".format(rotation_vector))
        #print ("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the frame.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        p0 = ( int(image_points[0][0]), int(image_points[0][1]) - 2)
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255,0,0), 2)
        angle = cal_angle(p0,p1,p2)
        print(angle)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(50) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
