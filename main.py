from flask import Flask, render_template, request, jsonify
import cv2
import os
import base64
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load models
faceProto = "data/opencv_face_detector.pbtxt"
faceModel = "data/opencv_face_detector_uint8.pb"
ageProto = "data/deploy_age.prototxt"
ageModel = "data/age_net.caffemodel"
genderProto = "data/deploy_gender.prototxt"
genderModel = "data/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def faceBox(faceNet, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
    return bboxs

def getGenderAge(face, genderNet, ageNet, MODEL_MEAN_VALUES, genderList, ageList):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]

    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]

    return gender, age

def read_image(filepath):
    try:
        # Try to read the image with OpenCV
        image = cv2.imread(filepath)
        if image is None:  # Fallback to Pillow for unsupported formats
            with Image.open(filepath) as img:
                img = img.convert("RGB")
                image = np.array(img)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        raise ValueError(f"Error reading the image file: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # For webcam capture
        if request.is_json and 'image' in request.json:
            image_data = request.json['image']
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            filepath = os.path.join("temp", "webcam_image.jpg")
            os.makedirs("temp", exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(image_bytes)

        # For file upload
        elif 'file' in request.files:
            file = request.files['file']
            if not file:
                return jsonify({"error": "No file uploaded"}), 400

            filepath = os.path.join("temp", file.filename)
            os.makedirs("temp", exist_ok=True)
            file.save(filepath)
        else:
            return jsonify({"error": "Invalid request format"}), 400

        # Process the image
        frame = read_image(filepath)
        if frame is None:
            return jsonify({"error": "Invalid image file"}), 400

        bboxs = faceBox(faceNet, frame)
        results = []
        for bbox in bboxs:
            face = frame[max(0, bbox[1]):min(bbox[3], frame.shape[0]),
                         max(0, bbox[0]):min(bbox[2], frame.shape[1])]
            gender, age = getGenderAge(face, genderNet, ageNet, MODEL_MEAN_VALUES, genderList, ageList)
            results.append({"gender": gender, "age": age, "bbox": bbox})

        os.remove(filepath)
        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

if __name__ == "__main__":
    app.run(debug=True)
