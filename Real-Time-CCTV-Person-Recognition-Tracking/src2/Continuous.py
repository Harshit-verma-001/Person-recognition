import datetime
import face_recognition
import cv2 
import numpy as np 
import os
import math
import sqlite3

camnum = 1
DB_PATH = os.path.join(os.path.dirname(__file__), "faces.db")

def load_known_faces():
    """Load known faces from SQLite database if available, otherwise known/ folder."""
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
            print(f"[WARN] Error loading from DB: {e}")

    # Fallback to local known/ directory if DB is empty
    known_dir = os.path.join(os.path.dirname(__file__), "known")
    if len(known_face_encodings) == 0 and os.path.exists(known_dir):
        for file_name in os.listdir(known_dir):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(known_dir, file_name)
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_face_encodings.append(encs[0])
                    known_face_names.append(os.path.splitext(file_name)[0])

    return known_face_encodings, known_face_names

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range_val = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_val * 2.0)
        return linear_val
    else:
        range_val = face_match_threshold
        linear_val = 1.0 - (face_distance / (range_val * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

known_face_encodings, known_face_names = load_known_faces()
print(f"[INFO] Loaded {len(known_face_names)} known identity(ies).")

video_capture = cv2.VideoCapture(0)

face_locations = [] 
face_encodings = [] 
face_names = []
process_this_frame = True
acclist = []

log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logfile_path = os.path.join(log_dir, "logfiles.txt")

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
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_idx = np.argmin(face_distances)

                if matches[best_match_idx]:
                    name = known_face_names[best_match_idx]
                    acc = round(1.0 - face_distances[best_match_idx], 4)

            face_names.append((name, acc))
    
    process_this_frame = not process_this_frame

    for (top, right, bottom, left), (name, acc) in zip(face_locations, face_names): 
        top *= 4 
        right *= 4 
        bottom *= 4 
        left *= 4

        font = cv2.FONT_HERSHEY_DUPLEX

        if name == "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"{name} ({acc})", (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            acclist.append(acc)

        log_msg = f"{name} (conf: {acc}) detected in camera {camnum} at {datetime.datetime.now()}\n"
        with open(logfile_path, "a") as f_log:
            f_log.write(log_msg)

        if name != "Unknown":
            person_log = os.path.join(log_dir, f"{name}.txt")
            with open(person_log, "a") as f_p:
                f_p.write(log_msg)
   
    cv2.imshow('Continuous Surveillance Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release() 
cv2.destroyAllWindows()
