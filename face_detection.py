import cv2

# Load the pre-trained face detector (comes with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

print("Face detection started! Press 'q' to quit.")

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (face detector works on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces — returns list of (x, y, w, h) for each face found
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # how much image is scaled down each step
        minNeighbors=5,    # how many neighbors each rectangle should have
        minSize=(30, 30)   # minimum face size to detect
    )

    # Draw a green rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add a label above the box
        cv2.putText(frame, "Face Detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show how many faces are detected
    cv2.putText(frame, f"Faces: {len(faces)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("AI Interviewer - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()