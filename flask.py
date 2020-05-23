
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

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

#alert threshold for eye
EYE_AR_THRESH = 0.30
#No. of frames after which alert will work
EYE_AR_CONSEC_FRAMES = 120

# initialize the frame count as well as a boolean used to
# indicate if the alarm is going off
count = 0
ALARM_ON = False

def cal_angle(a,b,c):
    v1 = np.array(a) - np.array(b)
    v0 = np.array(b) - np.array(c)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)
# initializing dlib's shape_predictor_68_face_landmarks.dat and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("starting video stream thread...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)
def model():
 while True:

    # convert frames to grayscale
    _,frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # when face not detected
    if(len(rects) == 0):
        count += 1
        print(count)

    else:

	    # loop over the face detections
	    for rect in rects:
	        # determine the facial landmarks for the face region, then
	        # convert the facial landmark (x, y)-coordinates to a NumPy
	        # array
	        shape = predictor(gray, rect)
	        shape = face_utils.shape_to_np(shape)

	        # for headpose
	        #2D image points. If you change the image, you need to change vector
	        image_points = np.array([
	                                    (shape[33, :]),     # Nose tip
	                                    (shape[8,  :]),     # Chin
	                                    (shape[36, :]),     # Left eye left corner
	                                    (shape[45, :]),     # Right eye right corne
	                                    (shape[48, :]),     # Left Mouth corner
	                                    (shape[54, :])      # Right mouth corner
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


			#for eye
	        # extract the left and right eye coordinates, then use the
	        # coordinates to compute the eye aspect ratio for both eyes
	        leftEye = shape[lStart:lEnd]
	        rightEye = shape[rStart:rEnd]
	        leftEAR = eye_aspect_ratio(leftEye)
	        rightEAR = eye_aspect_ratio(rightEye)

	        # average of eye aspect ratio
	        ear = (leftEAR + rightEAR) / 2.0

	        # visualize the convex hull for the left and right eye

	        leftEyeHull = cv2.convexHull(leftEye)
	        rightEyeHull = cv2.convexHull(rightEye)
	        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
	        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

	        #if eye aspect ratio is below the blink threshold increment the blink count
	        if (ear < EYE_AR_THRESH or angle < -70 or angle >70):
	            count += 1
	            print (count)
	            #print (ear)
	            print(angle)
	            if count >= EYE_AR_CONSEC_FRAMES:
                    flag=0

	                cv2.putText(frame, "!!!!FOCUS ON THE SCREEN!!!!", (10, 30),
	                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	        #the eye aspect ratio is not below the blink threshold, so reset the count
	        else:
                flag=1
	            count = 0

        #computed eye aspect ratio on the frame
        #cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(50) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

 # do a bit of cleanup
 cv2.destroyAllWindows()
 vs.release()

from flask import Flask, render_template, request
from flask_mysqldb import MySQL

app = Flask(_name_)


app.config['MYSQL_HOST'] = 'db4free.net'
app.config['MYSQL_USER'] = 'aashishraj'
app.config['MYSQL_PASSWORD'] = 'hello123'
app.config['MYSQL_DB'] = 'trackit_student'

mysql = MySQL(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        details = request.form
        
        cur = mysql.connection.cursor()
        if(flag==0):
            cur.execute("INSERT INTO trackit_student("Result") VALUES ("Alert")", (firstName, lastName))
        else:
            cur.execute("INSERT INTO trackit_student("Result") VALUES ("Non Alert")", (firstName, lastName))
        mysql.connection.commit()
        cur.close()
        return 'success'
    return render_template('index.html')


if _name_ == '_main_':
    app.run()
