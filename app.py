import cv2
from flask import Flask, make_response, request, Response
from flask_uploads import UploadSet, configure_uploads

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
photos = UploadSet('photos', ALLOWED_EXTENSIONS)
configure_uploads(app, (photos,))


def detect(path, hars):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hars_to_detect = hars.split(",")
    if "face" in hars_to_detect:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            if "eyes" in hars_to_detect:
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img


@app.route('/', methods=['POST'])
def hello():
    if request.method == 'POST' and 'photo' in request.files:
        hars = request.form.get('hars')
        if hars:
            print(hars)
        else:
            hars = "face"
        image = request.files['photo']
        filename = photos.save(image)
        img = detect("uploads/{}".format(filename), hars)
        retval, buffer = cv2.imencode('.png', img)
        response = make_response(buffer.tobytes())
        response.headers['content-type'] = "image/png"
        return response
    return Response("{'photo':'not sent'}", status=422, mimetype='application/json')


@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(name)


if __name__ == '__main__':
    app.run()

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
