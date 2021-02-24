# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>
from flask import Flask,request,jsonify, render_template, make_response
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import base64

app = Flask(__name__, template_folder="views")


@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/test',methods=['GET'])
def test():
    return "hello world!"

@app.route('/submit',methods=['POST'])
def submit():
    image = request.form['video_feed']
    # TODO: process the image as you see fit here to ensure the system recognizes
    # you and your teammates. Bonus points if you can prevent the system from being fooled by someone
    # holding up a photo of you or your teammates to the webcam, though this is not required.

    # For now, render the logged in page if the user is logged in.
    print(image)
    encoded_data = image.split(',')[1]
    im_bytes = base64.b64decode(encoded_data)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    print(im_arr)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_UNCHANGED)
    print(img.shape)
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    print(img.shape)
    print(np.nonzero(img))
    # imgdata = base64.b64decode(encoded_data)
    # with open("imageToSave.jpg", "wb") as fh:
    #     fh.write(imgdata)
    # img_data = base64.b64decode(img_data)
    # img_pil = Image.open(io.BytesIO(img_data))
    # rgb = cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2RGB)
    # print(rgb)
    # cv2.imwrite('color_img.png', rgb.astype(npuint8))
    # print("test")
    # print(type(rgb))
    if image:
        return render_template('logged_in.html')
    return render_template("unauthorized.html")
if __name__ == "__main__":
    app.run(debug=True)
