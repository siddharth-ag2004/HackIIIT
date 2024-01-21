def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    counter = 0
    id=-1
    bucket = storage.bucket()
    while True:
        success, img = cap.read()
        
        if not success:
            print("Error capturing frame from the camera.")
            continue

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace,tolerance=0.5)
            matchIndex = np.argmin(faceDis)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = int(y1 * 4), int(x2 * 4), int(y2 * 4), int(x1 * 4)
            bbox = (x1, y1, x2 - x1, y2 - y1)

            if matches[matchIndex]:
                color = (0, 255, 0)  # Green rectangle for a match
                id=studentIds[matchIndex]
                if counter==0:
                    counter=1
                   
            else:
                color = (0, 0, 255)  # Red square for no match
                # print("No match found")
        
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
        if counter!=0:
            if counter==1:
                studentinfo=db.reference('Students/'+id).get()
                 # Get the Image from the storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                print(studentinfo)
            counter+=1
            



        _, buffer = cv2.imencode('.jpg', img)
        imgencode = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + imgencode + b'\r\n')