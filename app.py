# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>
from flask import Flask,request,jsonify, render_template, make_response
from model_package import preprocessing

app = Flask(__name__, template_folder="views")


@app.route('/')
def home():
    return render_template('./auth.html')


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
    
    # Transforming the image into a numpy array that can be processed with OpenCV
    preprocessor = preprocessing.Preprocessor(image)
    encoded = preprocessor.get_base64()
    pixel_arr = preprocessor.decode_bytes(encoded)
    img = preprocessor.get_cv2_image(pixel_arr)
    
    print(img)

        
    if image:
        return render_template('logged_in.html')
    return render_template("unauthorized.html")
if __name__ == "__main__":
    app.run(debug=True)
