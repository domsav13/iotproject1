#! /usr/bin/python

# Import necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import os
import smtplib
import ssl
from email.message import EmailMessage

# -----------------------------
# Global Variables
# -----------------------------
TARGET_PERSON = ["Dominic"]  # Change this to the name of the person to detect
ENCODINGS_PATH = "encodings.pickle"
IMAGE_SAVE_PATH = "/home/pi/Desktop/Face_Images"
EMAIL_SENDER = "eyepicamera@gmail.com"
EMAIL_PASSWORD = "iqdb mqob quiy ludd" 
EMAIL_RECEIVER = ["dsavarino@gwmail.gwu.edu"]

# Ensure save directory exists
if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)

# -----------------------------
# Load Known Face Encodings
# -----------------------------
print("[INFO] Loading encodings + face detector...")
data = pickle.loads(open(ENCODINGS_PATH, "rb").read())

# Initialize Video Stream
print("[INFO] Starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)  # Allow camera to warm up

fps = FPS().start()

# -----------------------------
# Email Function
# -----------------------------
def send_email(image_path, detected_name):
    """Sends an email with the captured face image as an attachment and stops the program."""
    now = time.asctime()
    subject = f"Face Recognition Alert: {detected_name} Detected"

    body = f"""
    EyePi Camera detected {detected_name} on {now}.
    
    Please see the attached image.
    """

    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the captured image
    with open(image_path, "rb") as img_file:
        msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

    # Securely send the email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print(f"[INFO] Email sent successfully with {image_path}")
    print("[INFO] Stopping program...")
    exit()

# -----------------------------
# Facial Recognition Loop
# -----------------------------
currentname = "unknown"

while True:
    # Capture frame and resize for faster processing
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Detect faces in frame
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    # Loop over each detected face
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            if currentname != name:
                currentname = name
                print(f"[INFO] Detected: {currentname}")

        names.append(name)

    # Loop over recognized faces and display names
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

        # Check if the detected name matches the target
        if name == TARGET_PERSON:
            print(f"[INFO] {TARGET_PERSON} detected! Waiting 1 second before capturing image...")
            time.sleep(1)  

            # Save the image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_name = f"{TARGET_PERSON}_detected_{timestamp}.jpg"
            img_path = os.path.join(IMAGE_SAVE_PATH, img_name)
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Image saved: {img_path}")

            # Send email and stop the program
            send_email(img_path, TARGET_PERSON)

    # Display output window
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit on 'q' key press
    if key == ord("q"):
        break

    # Update FPS counter
    fps.update()

# Cleanup
fps.stop()
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
