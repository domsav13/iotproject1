import face_recognition
import imutils
import pickle
import time
import cv2
import os
import smtplib
import ssl
from email.message import EmailMessage
from imutils.video import VideoStream, FPS

# -----------------------------
# CONFIGURATION
# -----------------------------
TARGET_PERSON = "Dominic"  # Change to match stored name in encodings.pickle
ENCODINGS_PATH = "encodings.pickle"
IMAGE_SAVE_PATH = "/home/pi/Desktop/Face_Images"

EMAIL_SENDER = "eyepicamera@gmail.com"
EMAIL_PASSWORD = "iqdb mqob quiy ludd"  # Consider using environment variables
EMAIL_RECEIVER = ["dsavarino@gwmail.gwu.edu"]

# Ensure save directory exists
if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)

# -----------------------------
# LOAD ENCODINGS
# -----------------------------
print("[INFO] Loading known faces...")
data = pickle.loads(open(ENCODINGS_PATH, "rb").read())

# -----------------------------
# INITIALIZE VIDEO STREAM
# -----------------------------
print("[INFO] Starting video stream...")
vs = VideoStream(usePiCamera=True, resolution=(320, 240), framerate=30).start()
time.sleep(2.0)  # Allow camera to warm up
fps = FPS().start()

# -----------------------------
# EMAIL FUNCTION
# -----------------------------
def send_email(image_path):
    """Sends an email with the captured face image as an attachment and stops the program."""
    print(f"[INFO] Sending email with image: {image_path}")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file does not exist: {image_path}")
        return  # Prevent sending if file isn't found

    now = time.asctime()
    subject = f"Face Recognition Alert: {TARGET_PERSON} Detected"

    body = f"""
    EyePi Camera detected {TARGET_PERSON} on {now}.
    
    Please see the attached image.
    """

    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with open(image_path, "rb") as img_file:
            msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"[INFO] Email sent successfully with {image_path}")
        print("[INFO] Stopping program...")
        exit()  # Stop the program after sending the email
    
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

# -----------------------------
# FACE RECOGNITION LOOP
# -----------------------------
previously_detected = set()  # Stores previously detected names to avoid spamming prints

while True:
    # Grab a frame, resize for speed
    frame = vs.read()
    frame = imutils.resize(frame, width=300)

    # Detect faces & extract encodings
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    detected_names = set()  # Use a set to store detected names

    # Compare detected faces with known encodings
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for i, match in enumerate(matches) if match]
            name_counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                name_counts[name] = name_counts.get(name, 0) + 1

            name = max(name_counts, key=name_counts.get)

        detected_names.add(name)

    # Only print when a new person is detected
    if detected_names != previously_detected:
        print(f"[INFO] Detected Faces: {detected_names}")
        previously_detected = detected_names.copy()

    # If the target person is detected, take a picture and send an email
    if TARGET_PERSON in detected_names:
        print(f"[INFO] {TARGET_PERSON} detected! Capturing image in 1 second...")
        time.sleep(1)  # Small delay before capturing the image
        
        # Save the image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        img_name = f"{TARGET_PERSON}_detected_{timestamp}.jpg"
        img_path = os.path.join(IMAGE_SAVE_PATH, img_name)

        if cv2.imwrite(img_path, frame):
            print(f"[INFO] Image saved at {img_path}")
            send_email(img_path)  # Send the email and exit
        else:
            print(f"[ERROR] Failed to save image at {img_path}")

    # Show the video stream with detected names
    for ((top, right, bottom, left), name) in zip(boxes, detected_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Facial Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit if 'q' is pressed
    if key == ord("q"):
        break

    # Update FPS counter
    fps.update()

# -----------------------------
# CLEANUP
# -----------------------------
fps.stop()
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()

