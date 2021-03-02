# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>
from flask import Flask, request, render_template
from PIL import Image
import torch
from werkzeug.datastructures import IfRange
from model_package.preprocessing import Preprocessor
from model_package.model import Model

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
    
    # Transforming the image into a numpy array that can be processed with OpenCV
    preprocessor = Preprocessor(image)
    encoded = preprocessor.get_base64()
    pixel_arr = preprocessor.decode_bytes(encoded)
    img = preprocessor.get_cv2_image(pixel_arr)
    
    #print(img)
    

    model = Model(img)
    
    mtcnn = model.init_mtcnn()
    facenet = model.init_resnet()
    
    test = model.detect_human(mtcnn, img)

    if not test:
        print('triggered')
        return 'RecognizeMe only works with human faces, for now'
        
    
    else:
        test_vec_pos = model.file_to_vector(img, mtcnn, facenet)
        human_img = Image.open(open('data/harry_potter_1.jpg', mode='rb'))
        human_vec = model.file_to_vector(human_img, mtcnn, facenet)

        animal_img = Image.open(open('data/dog.jpg', mode='rb'))
        animal_vec = model.file_to_vector(animal_img, mtcnn, facenet)
        vec_dist_human = model.get_distance(test_vec_pos, human_vec)
        vec_dist_animal = model.get_distance(test_vec_pos, animal_vec)
    
        if vec_dist_human < vec_dist_animal:
            train_img = Image.open(open('data/mani.jpg', mode='rb'))
            test_img_neg = Image.open(open('data/harry_potter_1.jpg', mode='rb'))
                
            train_vec = model.file_to_vector(train_img, mtcnn, facenet)
            
            test_vec_pos = model.file_to_vector(img, mtcnn, facenet)
            test_vec_neg = model.file_to_vector(test_img_neg, mtcnn, facenet)
            
            vec_dist_pos = model.get_distance(train_vec, test_vec_pos)
            vec_dist_neg = model.get_distance(train_vec, test_vec_neg)
        else:
            print('Did not pass animal test')
            return 'RecognizeMe only works with human faces, for now'
            
    if vec_dist_pos < vec_dist_neg:
        return render_template('logged_in.html')
    return render_template("unauthorized.html")

if __name__ == "__main__":
    app.run(debug=True)
