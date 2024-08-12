import os
import pickle
import face_recognition
import numpy as np

def load_known_faces(database_path, encoding_file='encodings.pkl'):
    if os.path.exists(encoding_file):
        print("Loading encodings from file...")
        with open(encoding_file, 'rb') as file:
            known_face_encodings, known_face_names = pickle.load(file)
        print(f"Encodings loaded. Total known faces: {len(known_face_names)}")
    else:
        print("Encoding faces...")
        known_face_encodings, known_face_names = encode_faces_from_directory(database_path)
        print(f"Total known faces encoded: {len(known_face_names)}")

        print("Saving encodings to file...")
        with open(encoding_file, 'wb') as file:
            pickle.dump((known_face_encodings, known_face_names), file)
        print(f"Encodings saved to {encoding_file}")

    return known_face_encodings, known_face_names

def encode_faces_from_directory(directory):
    encodings = []
    names = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff')):
            image_path = os.path.join(directory, filename)
            print(f"Loading image: {image_path}")
            image = face_recognition.load_image_file(image_path)
            image_encodings = face_recognition.face_encodings(image)
            if image_encodings:
                for encoding in image_encodings:
                    encodings.append(encoding)
                    names.append(os.path.splitext(filename)[0])
                    print(f"Encoded face from {filename}")
    return encodings, names

def detect_faces(frame, known_face_encodings, known_face_names, accuracy_threshold):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    best_match_name = None
    best_match_accuracy = float('inf')

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if any(matches):
            best_match_index = np.argmin(distances)
            best_match_name = known_face_names[best_match_index]
            best_match_accuracy = distances[best_match_index]

            if best_match_accuracy < accuracy_threshold:
                return best_match_name, best_match_accuracy
    return None, None