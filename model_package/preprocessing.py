import numpy as np
import base64
import cv2

class Preprocessor:
    """
    Pipeline for preprocessing the base64 string into a numpy array
    that can be utilized with OpenCV
    """
    
    def __init__(self, data_uri: str):
        """
        The constructor for the Preprocessor class
        
        Parameters:
            encoded_data (str): the data URI for the image
        """
        self.data_uri = data_uri
    
    def get_base64(self) -> str:
        """
        Get the base64 encoded string from the data URI
        
        Returns:
            encoded_data (str): the base64 encoded string
        """
        encoded_data = self.data_uri.split(',')[1]
        
        return encoded_data
    
    def decode_bytes(self, encoded_data: str) -> str:
        """
        Get the byte data from the base64 string and decode the bytes
        
        Parameters:
            encoded_data (str): the base64 encoded string
            
        Returns:
            pixel_arr (ndarray): a 1 dimensional numpy array containing the pixel values of the image
        """
        
        pixel_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        
        return pixel_arr
    
    def get_cv2_image(self, pixel_arr: np.ndarray) -> np.ndarray:
        """
        Transforming the 1D pixel array into a 3D numpy array that can be processed with OpenCV
        
        Parameters:
            pixel_arr (ndarray): a 1 dimensional numpy array containing the pixel values of the image
        
        Returns:
            img (ndarray): a 3D numpy array that can be processed with OpenCV
        """
        img = cv2.imdecode(pixel_arr, cv2.IMREAD_COLOR)
        
        return img
