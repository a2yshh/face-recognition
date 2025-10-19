import cv2
import face_recognition
import numpy as np 
from deepface import DeepFace 
known_face_encodings = []
known_face_names = []
known_person1_image=face_recognition.load_image_file(r"C:\Users\ayush\OneDrive\Desktop\face-recognition\known-faces\Ayush1.jpeg")
known_person2_image=face_recognition.load_image_file(r"C:\Users\ayush\OneDrive\Desktop\face-recognition\known-faces\Shrestha.jpg")
known_person3_image=face_recognition.load_image_file(r"C:\Users\ayush\OneDrive\Desktop\face-recognition\known-faces\Roshan.jpg")
known_person4_image=face_recognition.load_image_file(r"C:\Users\ayush\OneDrive\Desktop\face-recognition\known-faces\Manmeet1.jpg")
known_person1_encoding=face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding=face_recognition.face_encodings(known_person2_image)[0]
known_person3_encoding=face_recognition.face_encodings(known_person3_image)[0]
known_person4_encoding=face_recognition.face_encodings(known_person4_image)[0]
known_face_encodings.append(known_person1_encoding)
known_face_encodings.append(known_person2_encoding)
known_face_encodings.append(known_person3_encoding)
known_face_encodings.append(known_person4_encoding)
known_face_names.append("ayush")
known_face_names.append("shrestha")
known_face_names.append("punpun")
known_face_names.append("manmeet")
video_capture = cv2.VideoCapture(0)
while True : 
    ref,frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame,face_locations) 
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = 0.0
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        confidence = (1 - face_distances[best_match_index]) * 100  # convert to %
        cv2.putText(frame, f"Confidence: {confidence:.2f}%", (left, bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255,0), 2)
        
    try :
        result = DeepFace.analyze(frame,actions=["emotion"],enforce_detection = False)

        if isinstance(result,list):
            dominant_emotion = result[0].get("dominant_emotion","unknown")
        else:
            result.get("dominant_emotion","unknown")
        cv2.putText(frame,f'Emotion:{dominant_emotion}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    except Exception as e :
        cv2.putText(frame,f'no emotion detected',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

