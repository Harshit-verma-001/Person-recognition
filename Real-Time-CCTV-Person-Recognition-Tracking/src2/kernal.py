import face_recognition
import cv2
import numpy as np
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "faces.db")

def load_known_faces():
    """Load known faces from SQLite database."""
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

    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()
video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
nameList = []

demo_file_path = os.path.join(os.path.dirname(__file__), "demofile.txt")

with open(demo_file_path, "w") as f:
    pass  # clear previous tracking points

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
            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_idx = np.argmin(face_distances)
                if face_distances[best_match_idx] <= 0.6:
                    name = known_face_names[best_match_idx]

            face_names.append(name)
   
    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        font = cv2.FONT_HERSHEY_DUPLEX

        if name not in nameList and name != "Unknown":
            nameList.append(name)
            with open(demo_file_path, "a") as f:
                f.write(f"{name} {left} {top} {right} {bottom}\n")

        if name == "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    cv2.imshow('Face Detection & Point Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Transition into tracking phase if targets were recorded
track_script = os.path.join(os.path.dirname(__file__), "track.py")
if os.path.exists(track_script) and len(nameList) > 0:
    print(f"[INFO] Launching tracker for targets: {', '.join(nameList)}")
    os.system(f'python "{track_script}"')
