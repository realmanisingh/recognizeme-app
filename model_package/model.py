import os

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import BayesianGaussianMixture

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


def knn(features: np.ndarray, labels: np.ndarray, n=5) -> KNeighborsClassifier:
    """
    Initialize a KNN classifier on the image data

    Parameters:
        features (numpy array): matrix where each row are the pixel values for an image
        labels (numpy array): vector where each value is the label for the corresponding row
        n (int): Number of neighbors to use by default for kneighbors queries.

    Returns:
        knn_model (KNeighborsClassifer): a scikit-learn knn classifer
    """
    labels = labels.reshape((labels.shape[0], 1))
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(features, labels)

    return model, le


def get_classifier(features, labels):
    gm = BayesianGaussianMixture(n_components=len(np.unique(labels)))

    labels = labels.reshape((labels.shape[0], 1))
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    gm.fit(features, labels)
    x = gm.predict_proba(features)

    return gm


class Classifier:
    def __init__(self, threshold=-1):
        self.distributions = {}
        self.labels = None
        self.threshold = threshold

    def fit(self, X, y):
        self.labels = np.unique(y)
        for label in self.labels:
            indices = y == label
            features_subset = X[indices]
            mean = np.mean(features_subset, axis=0)
            var = np.var(features_subset, axis=0)
            self.distributions[label] = (mean, var)
        return self

    def predict(self, X):
        probs = torch.zeros(len(self.labels), len(X))
        for i, label in enumerate(self.labels):
            mean, var = self.distributions[label]
            cov = torch.diag(torch.tensor(var))
            dist = MultivariateNormal(torch.tensor(mean).unsqueeze(0), cov.unsqueeze(0))
            probs[i] = dist.log_prob(torch.tensor(X))
        values, indices = torch.max(probs, dim=0)
        ood = values < self.threshold
        prediction = self.labels[indices.numpy()]
        prediction[ood.numpy()] = 'random'
        return prediction
