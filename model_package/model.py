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


def create_training_data(path: str) -> np.ndarray:
    """
    Create a feature matrix and label vectors for the images in the data
    directory
    
    Parameters:
        path (str): relative path of the data directory
    
    Returns:
        train_features (numpy array): matrix where each row is the pixel values for an image
        train_labels (numpy array): vector where each value is the label for the corresponding row
        in the matrix
    """
    labels = os.listdir(path)
    if '.DS_Store' in labels:
        labels.remove('.DS_Store')

    train_features = []
    train_labels = []
    for label in labels:
        label_path = f'{path}/{label}'
        images = os.listdir(label_path)

        for filename in images:
            np_arr = np.load(f'{label_path}/{filename}')
            train_labels.append(label)
            train_features.append(np_arr)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    print(train_features.shape)
    print(train_labels.shape)

    return train_features, train_labels
