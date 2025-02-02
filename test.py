import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model (Make sure the model path is correct)
try:
    model = load_model('model_file_50epochs.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize video capture (camera)
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Load the Haar cascade for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if faceDetect.empty():
    print("Error: Haar cascade XML file not found.")
    exit()
 
# Labels for different emotions
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Surprise', 6: 'Sad'}

# Start the video loop
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    for (x, y, w, h) in faces:
        sub_face_img = gray[y:y+h, x:x+w]  # Extract the face from the frame
        
        # Resize and normalize the face image
        resized = cv2.resize(sub_face_img, (48, 48))
        normalized = resized / 255.0
        
        # Reshape the image to fit the model input
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        
        # Predict the emotion
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        # Draw rectangles and display the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Draw face boundary
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)  # Highlight face
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)  # Background for label
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Emotion text

    # Display the frame
    cv2.imshow("Emotion Detection", frame)

    # Quit the video loop when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
