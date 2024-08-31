import cv2
import numpy as np

# Load the pre-trained model
model_path = "opencv_face_detector_uint8.pb"
config_path = "opencv_face_detector.pbtxt"

net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    h, w = frame.shape[:2]

    # Prepare the frame for the neural network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input to the neural network
    net.setInput(blob)

    # Perform forward pass to get the detections
    detections = net.forward()

    # Iterate over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections based on confidence threshold
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Draw the bounding box with confidence score
            text = f"{confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Face Detection with Confidence Scores', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
