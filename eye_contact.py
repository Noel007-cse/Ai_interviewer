import cv2
import mediapipe as mp
import numpy as np

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # gives more accurate eye landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (from MediaPipe's 468 points)
# Left eye corners
LEFT_EYE = [33, 160, 158, 133, 153, 144]
# Right eye corners
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Iris centers
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def get_eye_center(landmarks, eye_indices, frame_w, frame_h):
    """Get the center point of an eye"""
    points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
              for i in eye_indices]
    x = sum(p[0] for p in points) // len(points)
    y = sum(p[1] for p in points) // len(points)
    return (x, y)

def check_eye_contact(landmarks, frame_w, frame_h):
    """
    Check if person is looking at camera by comparing
    iris position relative to eye corners
    """
    # Get iris centers
    left_iris = get_eye_center(landmarks, LEFT_IRIS, frame_w, frame_h)
    right_iris = get_eye_center(landmarks, RIGHT_IRIS, frame_w, frame_h)

    # Get eye corners
    left_eye_left = (int(landmarks[33].x * frame_w), int(landmarks[33].y * frame_h))
    left_eye_right = (int(landmarks[133].x * frame_w), int(landmarks[133].y * frame_h))

    right_eye_left = (int(landmarks[362].x * frame_w), int(landmarks[362].y * frame_h))
    right_eye_right = (int(landmarks[263].x * frame_w), int(landmarks[263].y * frame_h))

    # Calculate iris position ratio (0 = far left, 1 = far right)
    left_eye_width = left_eye_right[0] - left_eye_left[0]
    right_eye_width = right_eye_right[0] - right_eye_left[0]

    if left_eye_width == 0 or right_eye_width == 0:
        return True  # can't calculate, assume ok

    left_ratio = (left_iris[0] - left_eye_left[0]) / left_eye_width
    right_ratio = (right_iris[0] - right_eye_left[0]) / right_eye_width

    avg_ratio = (left_ratio + right_ratio) / 2

    # If iris is roughly centered (0.35 to 0.65), person is looking at camera
    return 0.35 <= avg_ratio <= 0.65

# Open webcam
cap = cv2.VideoCapture(0)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Eye contact detection started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe needs RGB, OpenCV gives BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = face_mesh.process(rgb_frame)

    eye_contact = False
    status_color = (0, 0, 255)  # red by default

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * frame_w)
                y = int(landmarks[idx].y * frame_h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Draw iris
            for idx in LEFT_IRIS + RIGHT_IRIS:
                x = int(landmarks[idx].x * frame_w)
                y = int(landmarks[idx].y * frame_h)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            # Check eye contact
            eye_contact = check_eye_contact(landmarks, frame_w, frame_h)

    # Display status
    if eye_contact:
        status = "Eye Contact: GOOD ✓"
        status_color = (0, 255, 0)   # green
    else:
        status = "Eye Contact: LOOK AT CAMERA"
        status_color = (0, 0, 255)   # red

    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("AI Interviewer - Eye Contact", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()