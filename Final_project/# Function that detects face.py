# Function that detects face
def detect_face_and_eyes(image_path):
    # Load the pre-trained Haar Cascade classifier for face and eyes detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Read image
    image = cv2.imread(image_path)

    # Convert image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect face
    # face = face_cascade.detectMultiScale(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Initialize list for detected eyes
    eyes_detected = []

    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        image = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = gray[y:y+h,x:x+w]

        # In each face, detect eyes
        eyes = eye_cascade.detectMultiScale(faceROI)

        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            image = cv2.circle(image, eye_center, radius, (255, 0, 0), 4)
            eyes_detected.append((x + x2, y + y2, w2, h2))  # Append detected eye coordinates to the list

    # Display the image
    cv2.imshow('Capture - Face detection', image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    return image, faces, eyes_detected