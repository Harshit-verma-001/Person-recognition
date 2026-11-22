import sqlite3
import cv2
import numpy as np
import face_recognition

DB_PATH = "faces.db"

def load_database_faces():
    """Load all face encodings and names from SQLite database into memory."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("SELECT name, face_encoding FROM persons")
    rows = cursor.fetchall()
    conn.close()

    known_names = []
    known_encodings = []

    for name, encoding_bytes in rows:
        encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
        known_names.append(name)
        known_encodings.append(encoding)

    return known_names, known_encodings

def run_live_recognition(camera_index=0, distance_threshold=0.60):
    """
    Start webcam feed and match faces against database records in real time.
    Press 'q' to quit.
    """
    known_names, known_encodings = load_database_faces()

    if not known_names:
        print("[WARN] No faces found in database! Please add faces using 'manage_database.py' first.")
        print("Starting camera anyway (all faces will show as Unknown)...")
    else:
        print(f"[INFO] Loaded {len(known_names)} person(s) from database: {', '.join(set(known_names))}")

    print("[INFO] Opening camera feed... Press 'q' on the video window to exit.")

    video_capture = cv2.VideoCapture(camera_index)

    if not video_capture.isOpened():
        print(f"[ERROR] Cannot open camera at index {camera_index}. Please check webcam connections or permissions.")
        return

    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print("[ERROR] Failed to read frame from camera.")
            break

        # Process every alternate frame to improve FPS speed
        if process_this_frame:
            # Resize frame to 1/4 size for fast face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and face encodings in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"

                if len(known_encodings) > 0:
                    # Calculate Euclidean distances between live face and all DB faces
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if face_distances[best_match_index] < distance_threshold:
                        name = known_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Draw bounding boxes and labels on original frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale coordinates back up to full size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name != "Unknown":
                color = (0, 255, 0) # Green for matched person
            else:
                color = (0, 0, 255) # Red for unknown

            # Draw box around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw background box for label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            # Write person's name
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Show live output window
        cv2.imshow('Live CCTV Face Recognition', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera stopped.")

if __name__ == "__main__":
    run_live_recognition()
