import os

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(img, prob_threshold=0.2):
    """
    Get the vector embedding of a face from an image
    :param img: the image to detect a face
    :param prob_threshold: the probability threshold to say there are no faces in the image
    :return: boolean: whether there is a face, tensor: the vector embedding
    """
    try:
        img_cropped, prob = mtcnn(img, save_path=None, return_prob=True)
    except TypeError:
        return False, None
    if prob < prob_threshold:
        return False, None
    with torch.no_grad():
        img_embedding = resnet(img_cropped.unsqueeze(0))
    return True, img_embedding


def save_labeled_vec(vec: torch.Tensor, label: str, save_dir='./data'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    label_path = os.path.join(save_dir, label)
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    next_i = 0
    for file_name in os.listdir(label_path):
        i = int(os.path.splitext(file_name)[0])
        next_i = max(next_i, i)
    next_i += 1

    path = os.path.join(label_path, f'{next_i}.npy')
    np.save(path, vec.squeeze().numpy())
    print(f'saved vector to {path}')
