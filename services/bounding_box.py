import numpy as np


class BoundingBoxExtractor:
    def __init__(self):
        pass

    def from_segmentation_mask(self, mask):
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        return [x_min, y_min, width, height]

    def from_two_points(self, point1, point2):
        x_min = min(point1[0], point2[0])
        y_min = min(point1[1], point2[1])
        x_max = max(point1[0], point2[0])
        y_max = max(point1[1], point2[1])
        width = x_max - x_min
        height = y_max - y_min
        return [x_min, y_min, width, height]

    def to_coco_format(self, bbox, image_id=1, category_id=1, annotation_id=1):
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        }


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from sam_segmentation import SamSegmentation
    image_path = os.path.join(os.path.dirname(__file__), 'images', 'truck.jpg')
    sam = SamSegmentation()
    sam.set_image(image_path=image_path)
    input_point = [[500, 375]]
    input_label = [1]
    masks, scores, logits = sam.predict_segmentation_masks(
        input_point, input_label)

    bbox_extractor = BoundingBoxExtractor()

    for i, mask in enumerate(masks):
        bbox = bbox_extractor.from_segmentation_mask(mask)
        if bbox:
            coco_bbox = bbox_extractor.to_coco_format(
                bbox, image_id=1, category_id=1, annotation_id=i+1)
            print(f"COCO Bounding Box {i+1}: {coco_bbox}")

    # visualización de la imagen con bounding box a partir de la máscara
    for i, mask in enumerate(masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(sam.image)
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min,
                                              y_max - y_min, edgecolor='red', facecolor='none', linewidth=2))
        plt.title(f"Mask {i+1}, Score: {scores[i]:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    # A partir de dos puntos
    plt.figure(figsize=(10, 10))
    plt.imshow(sam.image)
    point1 = [500, 375]
    point2 = [700, 475]
    bbox = bbox_extractor.from_two_points(point1, point2)
    plt.gca().add_patch(plt.Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='red', facecolor='none', linewidth=2))
    plt.scatter([point1[0], point2[0]], [point1[1], point2[1]],
                color='red', marker='*', s=375, edgecolor='white', linewidth=1.25)
    plt.axis('off')
    plt.show()
