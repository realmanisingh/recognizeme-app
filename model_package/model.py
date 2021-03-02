import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(img):
    img_cropped = mtcnn(img, save_path=None)
    with torch.no_grad():
        img_embedding = resnet(img_cropped.unsqueeze(0))
    return img_embedding


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
