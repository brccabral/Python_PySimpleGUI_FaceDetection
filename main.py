import PySimpleGUI as sg
import cv2

layout = [
    [sg.Image("", key="-IMAGE-", background_color="white")],
    [
        sg.Text(
            "People in picture: 0", key="-TEXT-", justification="center", expand_x=True
        )
    ],
]

window = sg.Window("Face Detection", layout)

# get video
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    event, values = window.read(timeout=0)
    if event == sg.WIN_CLOSED:
        break

    _, frame = video.read()

    # detect faces
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        img_gray, scaleFactor=1.3, minNeighbors=7, minSize=(50, 50)
    )

    # update text
    window["-TEXT-"].update(f"People in picture: {len(faces)}")

    # draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # update the image
    img_bytes = cv2.imencode(".png", frame)[1].tobytes()
    window["-IMAGE-"].update(img_bytes)

window.close()
