import cv2
import dlib
import os
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Setup database
Base = declarative_base()
engine = create_engine('sqlite:///attendance.db')
Session = sessionmaker(bind=engine)
session = Session()

# Define attendance table
class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    timestamp = Column(DateTime)

Base.metadata.create_all(engine)

# Load face recognition models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def load_known_faces():
    known_encodings, known_names = [], []
    for file in os.listdir('known_faces'):  # Ensure this folder exists
        img = cv2.imread(f'known_faces/{file}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:
            shape = predictor(gray, faces[0])
            face_encoding = np.array(face_model.compute_face_descriptor(img, shape))
            known_encodings.append(face_encoding)
            known_names.append(file.split('.')[0])  # Employee name from filename
    return known_encodings, known_names

# Recognize faces in the frame
def recognize_faces_in_frame(frame, known_encodings, known_names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        face_encoding = np.array(face_model.compute_face_descriptor(frame, shape))

        matches = [np.linalg.norm(face_encoding - enc) < 0.6 for enc in known_encodings]
        if any(matches):
            match_idx = matches.index(True)
            name = known_names[match_idx]
        else:
            name = "Unknown"
        
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if name != "Unknown":
            mark_attendance(name)

    return frame

# Mark attendance in the database
def mark_attendance(name):
    attendance_record = Attendance(name=name, timestamp=datetime.now())
    session.add(attendance_record)
    session.commit()

# Load known faces
known_encodings, known_names = load_known_faces()

# Simple login
def login():
    username = input("Enter username: ")
    password = input("Enter password: ")
    if username == "Apoorva Rohilla" and password == "Apoorva@1234":  # Replace with secure method for production
        print("Access granted")
    else:
        print("Access denied")
        exit()

login()

# Start video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = recognize_faces_in_frame(frame, known_encodings, known_names)
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
