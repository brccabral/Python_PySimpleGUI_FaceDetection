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

    # update the image
    img_bytes = cv2.imencode(".png", frame)[1].tobytes()
    window["-IMAGE-"].update(img_bytes)

window.close()
