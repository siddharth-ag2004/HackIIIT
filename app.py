from flask import Flask, render_template, Response
import cv2
import face_recognition

app = Flask(__name__)

# Load known images and their encodings
known_images = []
known_encodings = []

known_images.append(face_recognition.load_image_file("./known_people/siddharth.jpeg"))
known_encodings.append(face_recognition.face_encodings(known_images[0])[0])

known_images.append(face_recognition.load_image_file("./known_people/bhaskar.jpeg"))
known_encodings.append(face_recognition.face_encodings(known_images[1])[0])

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize recognition status variables
siddharth_recognized = False
bhaskar_recognized = False

def generate_frames():
    global siddharth_recognized, bhaskar_recognized

    while True:
        success, frame = cap.read()

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Initialize face colors list with red (not recognized) for all faces
        face_colors = [(0, 0, 255)] * len(face_locations)

        # Check if any face is recognized
        for i, face_encoding in enumerate(face_encodings):
            # Compare the face with the known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.50)

            for j, match in enumerate(matches):
                if match:
                    face_colors[i] = (0, 255, 0)  # Green if recognized
                    if j == 0:  # Assuming Siddharth is the first known person
                        siddharth_recognized = True
                        bhaskar_recognized = False  # Reset other recognition status
                    elif j == 1:  # Assuming Bhaskar is the second known person
                        bhaskar_recognized = True
                        siddharth_recognized = False  # Reset other recognition status

        # Draw rectangles around each face with the determined colors
        for (top, right, bottom, left), color in zip(face_locations, face_colors):
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', siddharth_recognized=siddharth_recognized, bhaskar_recognized=bhaskar_recognized)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
