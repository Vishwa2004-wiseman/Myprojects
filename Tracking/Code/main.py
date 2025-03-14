import numpy as np
import cv2
import time

# Paths to model files
prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.2

# Class labels and colors for visualization
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")

# Start camera feed
print("Starting Camera Feed...")
vs = cv2.VideoCapture(0)  # Use 0 for the default camera
time.sleep(2.0)

while True:
    ret, frame = vs.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.resize(frame, (500, 500))
    (h, w) = frame.shape[:2]
    
    # Preprocess the image
    imResize = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(imResize, 0.007843, (300, 300), 127.5)
    
    # Forward pass through the model
    net.setInput(blob)
    detections = net.forward()
    
    # Process detections
    detShape = detections.shape[2]
    for i in np.arange(0, detShape):
        confidence = detections[0, 0, i, 2]
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
        break

# Cleanup
vs.release()
cv2.destroyAllWindows()
