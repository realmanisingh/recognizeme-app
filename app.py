# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>

from flask import Flask, request, render_template, redirect

from model_package.model import get_embedding, save_labeled_vec, create_training_data, Classifier
from model_package.preprocessing import Preprocessor

app = Flask(__name__, template_folder="views")

train_features, train_labels = create_training_data('data')
classifer = Classifier()
classifer.fit(train_features, train_labels)

@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/test', methods=['GET'])
def test():
    return "hello world!"


@app.route('/label')
def label():
    return render_template('./label.html')


@app.route('/submit_label', methods=['POST'])
def submit_label():
    image = request.form['video_feed']
    preprocessor = Preprocessor(image)
    encoded = preprocessor.get_base64()
    pixel_arr = preprocessor.decode_bytes(encoded)
    img = preprocessor.get_cv2_image(pixel_arr)
    success, vec = get_embedding(img)
    if success:
        save_labeled_vec(vec, request.form['label'])
    return redirect('/label')


@app.route('/submit', methods=['POST'])
def submit():
    image = request.form['video_feed']
    # TODO: process the image as you see fit here to ensure the system recognizes
    # you and your teammates. Bonus points if you can prevent the system from being fooled by someone
    # holding up a photo of you or your teammates to the webcam, though this is not required.
    # For now, render the logged in page if the user is logged in.

    # Transforming the image into a numpy array that can be processed with OpenCV
    preprocessor = Preprocessor(image)
    encoded = preprocessor.get_base64()
    pixel_arr = preprocessor.decode_bytes(encoded)
    img = preprocessor.get_cv2_image(pixel_arr)

    success, vec = get_embedding(img)
    if not success:
        print('No face detected!')
        return render_template('not_human.html')

    prediction = classifer.predict(vec.numpy())

    context = {'user': prediction[0]}

    if context['user'] != 'random':
        return render_template('logged_in.html', **context)
    return render_template("unauthorized.html")


if __name__ == "__main__":
    app.run(debug=True)
