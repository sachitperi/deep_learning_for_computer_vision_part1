# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained SMILE detector CNN")
ap.add_argument("-v", "--video",
                help="path to the optional video file")
args = vars(ap.parse_args())

# load the face detector cascade and SMILE detector CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# if a video path was not provided, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise load the video
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we didnot a frame then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, convert it to grayscale, and then clone the original frame
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # detect faces in the input frame and then clone the frame so that we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over thee face bounding box
    for (fx, fy, fw, fh) in rects:
        # extract the roi of faces from the grayscale image, resize it to a fixed 28x28 pixels, and the prepare the ROI for classification via CNN
        roi = gray[fy:fy+fh, fx:fx+fw]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine the probabilities for smiling and not smiling, then set the label accordingly
        (notsmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notsmiling else "Not Smiling"

        # dispaly the label and bounding box on the output frame
        cv2.putText(frameClone, label, (fx, fy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)

        # show our detected faces along with smiling/not smiling labels
        cv2.imshow("Frame", frameClone)

        # if 'q' key is pressed stop the loop
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

# cleanup thee camera and close any open windows
camera.release()
cv2.destroyAllWindows()