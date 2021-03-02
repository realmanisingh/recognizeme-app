from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def numpy_to_vector(img):
    img_cropped = mtcnn(img, save_path=None)
    with torch.no_grad():
        img_embedding = resnet(img_cropped.unsqueeze(0))
    return img_embedding

