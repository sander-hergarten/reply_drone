import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.v2 as T  # Use updated transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import json  # If using COCO-like annotations

# --- Configuration ---
DATASET_PATH = "path/to/your/dataset"  # Base path
ANNOTATIONS_FILE_TRAIN = os.path.join(
    DATASET_PATH, "train_annotations.json"
)  # Example COCO-style
ANNOTATIONS_FILE_VAL = os.path.join(
    DATASET_PATH, "val_annotations.json"
)  # Example COCO-style
IMAGE_DIR_TRAIN = os.path.join(DATASET_PATH, "train/images")
IMAGE_DIR_VAL = os.path.join(DATASET_PATH, "val/images")
MODEL_SAVE_PATH = "faster_rcnn_poster_detector.pth"
NUM_CLASSES = 2  # Background + Poster
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CONFIDENCE_THRESHOLD = 0.5  # Threshold for initial detection
MULTI_POSTER_PENALTY_FACTOR = (
    0.7  # Example: Reduce confidence by 30% if multiple posters
)


# --- Dataset Class (Example for COCO-like structure) ---
# You MUST adapt this to your specific dataset format and structure
class PosterDataset(Dataset):
    def __init__(self, image_dir, annotations_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        with open(annotations_file) as f:
            self.coco_data = json.load(f)  # Load COCO style data

        self.image_ids = [img["id"] for img in self.coco_data["images"]]
        self.img_id_to_filename = {
            img["id"]: img["file_name"] for img in self.coco_data["images"]
        }
        self.img_id_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        # Ensure class IDs start from 1 (0 is background)
        # Assuming your poster class ID in COCO is 1
        self.coco_id_to_contiguous_id = {1: 1}  # Map your poster category ID to 1

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, self.img_id_to_filename[img_id])
        img = Image.open(img_path).convert("RGB")

        annotations = self.img_id_to_anns.get(img_id, [])
        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ann in annotations:
            # Map COCO category ID to our contiguous ID (1 for poster)
            coco_cat_id = ann["category_id"]
            if coco_cat_id in self.coco_id_to_contiguous_id:
                label = self.coco_id_to_contiguous_id[coco_cat_id]
            else:
                continue  # Skip if not the poster class

            # COCO format: [x_min, y_min, width, height]
            # Torchvision format: [x_min, y_min, x_max, y_max]
            xmin, ymin, w, h = ann["bbox"]
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            area.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])  # Use the original COCO image ID

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            # Note: transforms.v2 handles image and targets together
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_ids)


# --- Transforms ---
def get_transform(train):
    transforms = []
    # Converts the image, maps the values to [0..1] and channels to CxHxW
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        # Add data augmentation here if needed
        transforms.append(T.RandomHorizontalFlip(0.5))
        # Add more augmentations like color jitter, rotation etc.
        # transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # Sanitize bounding boxes is important after augmentations
    transforms.append(T.SanitizeBoundingBoxes())
    return T.Compose(transforms)


# --- Model Definition ---
def get_model_instance_segmentation(num_classes):
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# --- Utility Functions ---
def collate_fn(batch):
    return tuple(zip(*batch))


# --- Training Function ---
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    metric_logger = torchvision.utils.misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", torchvision.utils.misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"

    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = torchvision.utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


# --- Main Fine-tuning Script ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Datasets
    print("Loading datasets...")
    dataset_train = PosterDataset(
        IMAGE_DIR_TRAIN, ANNOTATIONS_FILE_TRAIN, get_transform(train=True)
    )
    dataset_val = PosterDataset(
        IMAGE_DIR_VAL, ANNOTATIONS_FILE_VAL, get_transform(train=False)
    )

    # DataLoaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Adjust num_workers
        collate_fn=collate_fn,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,  # Adjust num_workers
        collate_fn=collate_fn,
    )
    print("Datasets loaded.")

    # Model
    print("Loading model...")
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)
    print("Model loaded.")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(
            model, optimizer, data_loader_train, DEVICE, epoch, print_freq=50
        )

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the validation set (optional, add evaluation function)
        # evaluate(model, data_loader_val, device=DEVICE) # You need to implement evaluate()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed.")

    # Save the trained model
    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Training finished.")


# --- Inference Function (Example) ---
def run_inference_with_custom_confidence(
    model_path, image_path, device, confidence_threshold, penalty_factor
):
    # Load the trained model
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Load and transform the image
    img = Image.open(image_path).convert("RGB")
    transform = get_transform(train=False)  # Use validation transforms
    # Need to unsqueeze(0) to add batch dimension
    input_tensor, _ = transform(img, None)  # Transform only the image for inference
    input_batch = [input_tensor.to(device)]

    with torch.no_grad():
        outputs = model(input_batch)

    # Process outputs
    # outputs[0] contains {'boxes': tensor, 'labels': tensor, 'scores': tensor}
    pred_scores = outputs[0]["scores"].cpu().numpy()
    pred_boxes = outputs[0]["boxes"].cpu().numpy()
    pred_labels = outputs[0]["labels"].cpu().numpy()

    # Filter by confidence threshold
    keep_indices = np.where(pred_scores >= confidence_threshold)[0]
    filtered_scores = pred_scores[keep_indices]
    filtered_boxes = pred_boxes[keep_indices]
    filtered_labels = pred_labels[keep_indices]  # Should mostly be 1 (poster)

    num_detections = len(filtered_scores)
    final_results = []

    if num_detections == 0:
        print("No posters detected with sufficient confidence.")
        # You could return a default low confidence here if needed
        # final_results.append({'box': None, 'label': 'poster', 'confidence': 0.05})
    elif num_detections == 1:
        print(f"1 poster detected.")
        # Confidence is assumed to correlate with visibility
        final_results.append(
            {
                "box": filtered_boxes[0].tolist(),
                "label": "poster",  # Assuming label 1 is poster
                "confidence": float(filtered_scores[0]),
            }
        )
    else:  # More than 1 poster detected
        print(f"{num_detections} posters detected. Applying penalty.")
        for i in range(num_detections):
            penalized_confidence = filtered_scores[i] * penalty_factor
            final_results.append(
                {
                    "box": filtered_boxes[i].tolist(),
                    "label": "poster",
                    "confidence": float(penalized_confidence),
                }
            )

    return final_results


# --- Example Inference Usage ---

# if __name__ == "__main__":
#     # Ensure model is trained and saved before running inference
#     if os.path.exists(MODEL_SAVE_PATH):
#         test_image = 'path/to/your/test_image.jpg' # Provide a test image path
#         if os.path.exists(test_image):
#             results = run_inference_with_custom_confidence(
#                 MODEL_SAVE_PATH,
#                 test_image,
#                 DEVICE,
#                 CONFIDENCE_THRESHOLD,
#                 MULTI_POSTER_PENALTY_FACTOR
#             )
#             print("\nInference Results:")
#             for res in results:
#                 print(f"  Box: {res['box']}, Label: {res['label']}, Confidence: {res['confidence']:.4f}")
#         else:
#             print(f"Test image not found: {test_image}")
#     else:
#         print(f"Trained model not found: {MODEL_SAVE_PATH}. Please train the model first.")
