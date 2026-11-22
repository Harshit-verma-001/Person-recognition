import os
import sqlite3
import numpy as np

try:
    import face_recognition
except ImportError:
    face_recognition = None

DB_PATH = "faces.db"

def init_db():
    """Create Database Table if it doesn't exist."""
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
    conn.commit()
    conn.close()

def clear_database():
    """Remove ALL stored faces from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM persons")
    conn.commit()
    count = cursor.rowcount
    conn.close()
    print(f"[OK] Cleared database! Removed all {count} stored face record(s).")
    return count

def delete_person(name):
    """Delete a specific person by name from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM persons WHERE LOWER(name) = LOWER(?)", (name,))
    conn.commit()
    count = cursor.rowcount
    conn.close()
    if count > 0:
        print(f"[OK] Removed '{name}' ({count} encoding(s)) from database.")
    else:
        print(f"[WARN] No records found for person named '{name}'.")
    return count

def add_person(name, image_path):
    """Extract face encoding from image file and store in database."""
    init_db()
    if face_recognition is None:
        print("[ERROR] 'face_recognition' library is not installed. Please run: pip install face_recognition opencv-python")
        return False

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return False

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"[ERROR] No face detected in image: {image_path}")
        return False

    # Store encoding bytes in DB
    encoding_bytes = encodings[0].tobytes()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO persons (name, face_encoding) VALUES (?, ?)", (name, encoding_bytes))
    conn.commit()
    conn.close()
    print(f"[OK] Successfully added '{name}' using image '{image_path}'!")
    return True

def bulk_import_folder(folder_path):
    """
    Import all images from a directory.
    Filenames become person names (e.g. 'John_Doe.jpg' -> Name: 'John Doe').
    """
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return 0

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    added_count = 0

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):
            person_name = os.path.splitext(file_name)[0].replace('_', ' ').title()
            full_path = os.path.join(folder_path, file_name)
            print(f"Processing '{file_name}' for person '{person_name}'...")
            if add_person(person_name, full_path):
                added_count += 1

    print(f"\n[DONE] Bulk import complete! Total added: {added_count} person(s).")
    return added_count

def list_persons():
    """Display all registered people in database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, created_at FROM persons")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[INFO] Database is currently empty.")
        return []

    print("\n------------------ REGISTERED PERSONS ------------------")
    print(f"{'ID':<6} | {'Name':<25} | {'Registered At'}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:<6} | {row[1]:<25} | {row[2]}")
    print("-" * 60)
    print(f"Total: {len(rows)} record(s)\n")
    return rows

def enroll_via_webcam(name, max_samples=5):
    """
    Open webcam and capture multiple facial gestures/angles for a person.
    Saves all captured encodings under the same name in database.
    """
    init_db()
    if face_recognition is None:
        print("[ERROR] 'face_recognition' library is not installed.")
        return 0

    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV ('cv2') is not installed.")
        return 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam for gesture enrollment.")
        return 0

    print(f"\n[INFO] Starting Webcam Gesture Enrollment for '{name}'...")
    print("Instructions:")
    print(" - Look at camera and change gestures (smile, tilt head left/right, raise eyebrows).")
    print(" - Press SPACEBAR to capture a face gesture.")
    print(" - Press 'q' when finished.\n")

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Person: {name} | Samples: {saved_count}/{max_samples}",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to Capture Gesture | Press Q to Exit",
                    (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Webcam Gesture Enrollment", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, locations)

            if len(encodings) == 0:
                print("[WARN] No face detected in current frame. Please face the camera and try again.")
            else:
                encoding_bytes = encodings[0].tobytes()
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO persons (name, face_encoding) VALUES (?, ?)", (name, encoding_bytes))
                conn.commit()
                conn.close()
                saved_count += 1
                print(f"[OK] Captured gesture sample {saved_count}/{max_samples} for '{name}'!")

                if saved_count >= max_samples:
                    print(f"[DONE] Captured all {max_samples} gesture samples for '{name}'!")
                    break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_count

def interactive_menu():
    """Interactive command-line interface for managing the face database."""
    init_db()
    while True:
        print("\n================ FACE DATABASE MANAGER ================")
        print("1. View all registered persons")
        print("2. Add a new person (Single Image File)")
        print("3. Capture Multi-Gesture Samples via Webcam")
        print("4. Bulk import photos from a folder")
        print("5. Delete a specific person by name")
        print("6. Clear ALL records (Empty Database)")
        print("7. Start Live Camera Verification")
        print("8. Exit")
        print("=======================================================")
        choice = input("Select an option (1-8): ").strip()

        if choice == '1':
            list_persons()
        elif choice == '2':
            name = input("Enter person's name: ").strip()
            path = input("Enter image file path: ").strip().strip('"')
            add_person(name, path)
        elif choice == '3':
            name = input("Enter person's name for webcam enrollment: ").strip()
            if name:
                enroll_via_webcam(name)
        elif choice == '4':
            folder = input("Enter folder path containing images: ").strip().strip('"')
            bulk_import_folder(folder)
        elif choice == '5':
            name = input("Enter name of person to delete: ").strip()
            delete_person(name)
        elif choice == '6':
            confirm = input("[WARN] Are you sure you want to delete ALL face records? (yes/no): ").strip().lower()
            if confirm in ('yes', 'y'):
                clear_database()
            else:
                print("Operation cancelled.")
        elif choice == '7':
            try:
                from recognize_live import run_live_recognition
                run_live_recognition()
            except Exception as e:
                print(f"[ERROR] Could not start live camera recognition: {e}")
        elif choice == '8':
            print("Exiting Face Database Manager. Goodbye!")
            break
        else:
            print("Invalid selection. Please enter a number from 1 to 8.")

if __name__ == "__main__":
    interactive_menu()
