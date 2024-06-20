import sys
sys.path.append("./")
from utils.download_file import download_file
from utils.device_available import device_available
from segment_anything import sam_model_registry, SamPredictor
from skimage.measure import centroid
import numpy as np
import cv2
import os

# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


class SamSegmentation:
    def __init__(self, model_type='vit_h', sam_checkpoint='sam_vit_h_4b8939.pth'):
        self.device = device_available()
        self.model_type = model_type
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(
            os.path.join(models_path, sam_checkpoint)
        ):
            os.makedirs(models_path, exist_ok=True)
            print(f"Downloading {sam_checkpoint}...")
            download_file(
                f"https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}", sam_checkpoint, models_path)
        self.sam_checkpoint = os.path.join(models_path, sam_checkpoint)
        self.sam = sam_model_registry[self.model_type](self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image_path='', image=None):
        """This function sets the image to be used for segmentation.

        Args:
            image_path (str): The path to the image file.
        """
        if image is None:
            self.image = cv2.imread(image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            self.image = image

    def predict_segmentation_masks(self, input_point, input_label):
        """This function predicts the segmentation masks for the input point and label.

        Args:
            input_point (list): list of points example [[x1, y1], [x2, y2], ...]
            input_label (list): list of labels example [0, 1, 0, ...]

        Returns:
            masks: An ndarray of shape (N, H, W) where N is the number of masks, H is the height of the mask, and W is the width of the mask.
            scores: An ndarray of shape (N,) where N is the number of masks.
            logits: An ndarray of shape (N, H, W) where N is the number of masks, H is the height of the mask, and W is the width of the mask.
        """
        self.predictor.set_image(self.image)
        masks, scores, logits = self.predictor.predict(
            np.array(input_point),
            np.array(input_label)
        )
        return masks, scores, logits

    def calculate_centroid(self, mask=None):
        """This function calculates the centroid of the mask.

        Args:
            mask (ndarray): An ndarray of shape (H, W) where H is the height of the mask, and W is the width of the mask.

        Returns:
            centroid: An ndarray of shape (2,) where the first element is the y-coordinate and the second element is the x-coordinate of the centroid.
        """
        if mask is None and hasattr(self, 'masks'):
            mask = self.masks[0]
        elif mask is None:
            raise ValueError('mask is required')
        return centroid(mask)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # It require to have the image truck.jpg in the folder images
    image_path = os.path.join(os.path.dirname(__file__), 'images', 'truck.jpg')
    sam = SamSegmentation()
    sam.set_image(image_path=image_path)
    input_point = [[500, 375]]
    input_label = [1]
    masks, scores, logits = sam.predict_segmentation_masks(
        input_point, input_label)

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate(
                [np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
                   marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
                   marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(sam.image)
        show_mask(mask, plt.gca())
        centroid_point = sam.calculate_centroid(mask)
        show_points(np.array(input_point), np.array(input_label), plt.gca())
        plt.scatter(centroid_point[1], centroid_point[0],
                    color='blue', marker='*', s=375, edgecolor='white', linewidth=1.25)
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
