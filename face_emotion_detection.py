import cv2
import numpy as np
from keras.models import load_model

# Load model
emotion_model = load_model('fixed_model.h5')

# Emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def detect_emotions():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract and resize face to 64x64 (model's expected input)
            face_roi = gray[y:y+h, x:x+w]
            resized = cv2.resize(face_roi, (64, 64))
            
            # Normalize and reshape for model
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 64, 64, 1))
            
            # Predict emotion
            predictions = emotion_model.predict(reshaped)
            emotion = EMOTIONS[np.argmax(predictions)]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions()