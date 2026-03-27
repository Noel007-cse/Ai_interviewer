import cv2

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

print("Camera started! Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If frame was read successfully
    if not ret:
        print("Failed to grab frame")
        break

    # Show the frame in a window
    cv2.imshow("AI Interviewer - Camera Test", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()