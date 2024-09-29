import cvzone
import pickle
from cvzone.FaceMeshModule import FaceMeshDetector
import warnings
import cv2
import mediapipe as mp
import time
import numpy as np
from math import hypot

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Face and Hand Detection classes
class FaceBehaviourDetector:
    def __init__(self, model_path='model.pkl'):
        self.detector = FaceMeshDetector(maxFaces=1)
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def detect_and_predict(self, frame):
        img, faces = self.detector.findFaceMesh(frame)
        mood = "Unknown"
        if faces:
            face = faces[0]
            face_data = list(np.array(face).flatten())
            try:
                result = self.model.predict([face_data])
                mood = str(result[0])  # happy/sad etc.
            except Exception:
                pass
        return mood

class HandSwipeDetector:
    def __init__(self, velocity_threshold=20, middle_tolerance=50):
        self.prev_x = None
        self.prev_time = None
        self.velocity_threshold = velocity_threshold
        self.middle_tolerance = middle_tolerance
        self.swipe_detected = False
        self.middle_position = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, model_complexity=1, 
                                         min_detection_confidence=0.75, min_tracking_confidence=0.75, 
                                         max_num_hands=1)
        self.draw = mp.solutions.drawing_utils

    def detect_swipe(self, landmarkList):
        if len(landmarkList) > 0:
            current_x = landmarkList[0][1]
            current_time = time.time()

            if self.prev_x is not None and self.prev_time is not None:
                velocity = (current_x - self.prev_x) / (current_time - self.prev_time)
                
                if self.swipe_detected:
                    if abs(current_x - self.middle_position) < self.middle_tolerance:
                        self.swipe_detected = False
                    return "state"

                if velocity < -self.velocity_threshold and current_x < self.middle_position - self.middle_tolerance:
                    self.swipe_detected = True
                    return "left"
                elif velocity > self.velocity_threshold and current_x > self.middle_position + self.middle_tolerance:
                    self.swipe_detected = True
                    return "right"

            self.prev_x, self.prev_time = current_x, current_time

        return "state"

    def detect_hands(self, frame):
        Process = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarkList = []
        if Process.multi_hand_landmarks:
            handlm = Process.multi_hand_landmarks[0]
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])
            self.draw.draw_landmarks(frame, handlm, self.mp_hands.HAND_CONNECTIONS)
        return landmarkList, frame

    def calculate_thumb_index_distance(self, landmarkList, frame):
        if len(landmarkList) > 8:
            x_1, y_1 = landmarkList[4][1], landmarkList[4][2]
            x_2, y_2 = landmarkList[8][1], landmarkList[8][2]
            cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)
            distance = hypot(x_2 - x_1, y_2 - y_1)
            return distance
        return None

# Main class to handle the application logic
class HandFaceInteraction:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.hand_detector = HandSwipeDetector()
        self.face_detector = FaceBehaviourDetector()

    def run(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error reading from webcam.")
            return None, None, None

        frame = cv2.flip(frame, 1)
        if self.hand_detector.middle_position is None:
            self.hand_detector.middle_position = frame.shape[1] // 2

        # Hand Detection
        landmarkList, frame = self.hand_detector.detect_hands(frame)
        swipe_result = self.hand_detector.detect_swipe(landmarkList)

        # Calculate thumb-index distance
        distance = self.hand_detector.calculate_thumb_index_distance(landmarkList, frame)

        # Face Detection
        mood = self.face_detector.detect_and_predict(frame)

        return swipe_result, distance, mood, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandFaceInteraction()

    while True:
        swipe_result, distance, mood, frame = app.run()

        # Check if data was successfully processed
        if frame is None:
            break

        # Prepare text to display
        overlay_text = []

        if swipe_result == "left":
            print("Left detected!")
        elif swipe_result == "right":
            print("Right detected!")

        if distance is not None:
            overlay_text.append(f"Distance: {distance:.2f}")

        if mood != "Unknown":
            overlay_text.append(f"Mood: {mood}")

        # Overlay text on the frame
        for i, text in enumerate(overlay_text):
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    app.release()
