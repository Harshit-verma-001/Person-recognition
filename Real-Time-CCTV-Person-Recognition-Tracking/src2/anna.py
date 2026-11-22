import datetime
import face_recognition
import cv2 
import numpy as np 
import smtplib
import ssl
import os
import math
import sqlite3

camnum = 1
counterr = 0
DB_PATH = os.path.join(os.path.dirname(__file__), "faces.db")

def notif(detected_person="Unknown"):
    """Send security alert notification when an unknown person persists in feed."""
    sender_email = os.getenv("ALERT_SENDER_EMAIL")
    receiver_email = os.getenv("ALERT_RECEIVER_EMAIL")
    password = os.getenv("ALERT_EMAIL_PASSWORD")

    if not (sender_email and receiver_email and password):
        print(f"[ALERT] Security Alert: {detected_person} detected! (Configure ALERT_* env vars for email delivery)")
        return

    port = 465
    smtp_server = "smtp.gmail.com"
    message = f"""\
Subject: Security Alert - Intruder Detected

An unrecognized individual was detected on camera {camnum} at {datetime.datetime.now()}.
"""
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
        print("[INFO] Security alert email delivered.")
    except Exception as e:
        print(f"[WARN] Failed to send email alert: {e}")

def load_known_faces():
    """Load known identities from database."""
    known_face_encodings = []
    known_face_names = []

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name, face_encoding FROM persons")
            rows = cursor.fetchall()
            conn.close()
            for name, encoding_bytes in rows:
                encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
                known_face_names.append(name)
                known_face_encodings.append(encoding)
        except Exception as e:
            print(f"[WARN] Database load failed: {e}")

    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()
video_capture = cv2.VideoCapture(0)

face_locations = [] 
face_encodings = [] 
face_names = []
process_this_frame = True
acclist = []

while True:
    ret, frame = video_capture.read()
    if not ret or frame is None:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) 
        face_names = []

        for face_encoding in face_encodings:
            name = "Unknown"
            acc = 0.0

            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_idx = np.argmin(face_distances)
                if face_distances[best_idx] <= 0.6:
                    name = known_face_names[best_idx]
                    acc = round(float(1.0 - face_distances[best_idx]), 4)

            face_names.append((name, acc))
    
    process_this_frame = not process_this_frame

    for (top, right, bottom, left), (name, acc) in zip(face_locations, face_names): 
        top *= 4 
        right *= 4 
        bottom *= 4 
        left *= 4

        font = cv2.FONT_HERSHEY_DUPLEX
        if name == "Unknown":
            counterr += 1
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"{name} {acc}", (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            acclist.append(acc)

        with open('logfiles.txt', 'a') as the_file:
            the_file.write(f"{name} with score {acc} detected in camera {camnum} at {datetime.datetime.now()}\n")

    if counterr > 15:
        notif("Unknown Individual")
        counterr = 0

    cv2.imshow('Alert & Monitoring Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release() 
cv2.destroyAllWindows()
