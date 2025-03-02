import cv2
import time
import os
import smtplib
import ssl
from email.message import EmailMessage

# Global variable: Specify the object to detect
DETECTED_OBJECT = "person"  # Change this to the object you want to detect

# Object Detection Configuration
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Directory to save images
save_path = "/home/pi/Desktop/Images"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Email Configuration
email_sender = "eyepicamera@gmail.com"
email_password = "iqdb mqob quiy ludd"  # Consider using environment variables for security
email_receiver = ["dsavarino@gwmail.gwu.edu"]

def send_email(image_path):
    """Sends an email with the captured image as an attachment and stops the program."""
    now = time.asctime()
    subject = f"{DETECTED_OBJECT.capitalize()} Detected - EyePi Camera"
    
    body = f"""
    EyePi Camera detected a {DETECTED_OBJECT} on {now}.
    
    Please see the attached image.
    """
    
    msg = EmailMessage()
    msg["From"] = email_sender
    msg["To"] = email_receiver
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach image
    with open(image_path, "rb") as img_file:
        msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

    # Securely send the email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(email_sender, email_password)
        server.send_message(msg)
    
    print(f"Email sent successfully with {image_path}")
    print("Stopping program...")
    exit()  # Stop the program after sending the email

def getObjects(img, thres, nms, draw=True, objects=[]):
    """Detects specified objects in an image and returns detection results."""
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    object_detected = False

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                if className == DETECTED_OBJECT:
                    object_detected = True

    return img, objectInfo, object_detected


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result, objectInfo, object_detected = getObjects(img, 0.7, 0.2, objects=[DETECTED_OBJECT])  

        if object_detected:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_name = f"{DETECTED_OBJECT}_detected_{timestamp}.jpg"
            img_path = os.path.join(save_path, img_name)
            cv2.imwrite(img_path, img)
            print(f"Image saved: {img_path}")

            # Send email with the captured image and stop the program
            send_email(img_path)

        cv2.imshow("Output", img)
        cv2.waitKey(1)
