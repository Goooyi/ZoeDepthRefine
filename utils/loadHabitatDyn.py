import cv2

def load_HabitatDyn(dataset, idx, buff_size):
    if idx + buff_size > len(dataset):
        return None
    else: