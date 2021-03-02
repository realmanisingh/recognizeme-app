from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import torch
import cv2
import numpy as np

#@title Image file names
#train_image = 'data/harry_potter_1.jpg' #@param {type: "string"}
#test_postitive_image = 'data/harry_potter_2.jpg' #@param {type: "string"}
#test_negative_image = 'data/hermione_1.jpg' #@param {type: "string"}

class Model:
    """
    The Facenet + MTCNN models that will be used for facial recognition
    """
    
    def __init__(self, input_img: np.ndarray):
        self.input_img = input_img
        

    def init_mtcnn(self):
        """
        Instantiate the MTCNN model
        """
        mtcnn = MTCNN()
    
        return mtcnn

    def init_resnet(self):
        """
        Instantiate Resnet model
        """
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
        return resnet

    def file_to_vector(self, img: np.ndarray, mtcnn: MTCNN, resnet: InceptionResnetV1):
        """
        Transform numpy array representation of image into an embedding
        """
        img_cropped = mtcnn(img, save_path=None)
        with torch.no_grad():
            img_embedding = resnet(img_cropped.unsqueeze(0))
        return img_embedding
    
    def get_distance(self, img_1: np.ndarray, img_2: np.ndarray):
        """
        Calculate distance between two image embeddings
        """
        
        return torch.dist(img_1, img_2)
    
    def detect_human(self, mtcnn, img):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))     
        # Detect faces
        boxes, _ = mtcnn.detect(img)
        
        print(boxes)
        #print(tracked_frame)
        
        if type(boxes) == None:
            print('went to if')
            return False
        else:
            return True
        


# train_vec = file_to_vector(train_image)
# train_vec.shape

# test_vec_pos = file_to_vector(test_postitive_image)
# test_vec_neg = file_to_vector(test_negative_image)

# vec_dist_pos = torch.dist(train_vec, test_vec_pos)
# vec_dist_neg = torch.dist(train_vec, test_vec_neg)
# vec_dist_pos, vec_dist_neg

# print(vec_dist_pos, vec_dist_neg)