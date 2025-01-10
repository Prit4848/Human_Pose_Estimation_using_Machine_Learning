import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile


# Constants
DEMO_IMAGE = 'stand.jpg'
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Model and Input Configurations
width, height = 368, 368
inWidth, inHeight = width, height
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

st.title("Human Pose Estimation OpenCV")
st.text('Ensure a clear image or video for better pose estimation.')

# File uploader for image and video
option = st.radio("Choose Input Type", ["Image", "Video", "Camera"])
img_file_buffer = None
video_file_buffer = None
camera = None

if option == "Image":
    img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
elif option == "Video":
    video_file_buffer = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
elif option == "Camera":
    st.warning("Press Start to enable the camera.")
    camera = st.button("Start Camera")

thres = st.slider('Threshold for detecting the key points', min_value=0, value=20, max_value=100, step=5) / 100

@st.cache
def poseDetector(frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    points = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x, y = (frameWidth * point[0]) / out.shape[3], (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair[0], pair[1]
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

if option == "Image" and img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    output = poseDetector(image)
    st.image(output, caption="Pose Estimation on Image", use_column_width=True)

elif option == "Video" and video_file_buffer is not None:
    # Use tempfile to store the video file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file_buffer.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = poseDetector(frame)
        stframe.image(output, channels="BGR", use_column_width=True)
    cap.release()

elif option == "Camera" and camera:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output = poseDetector(frame)
        stframe.image(output, channels="BGR", use_column_width=True)
    cap.release()

st.markdown("### Ensure OpenCV and TensorFlow models are correctly configured!")
