"""
Fine-tuning Script for Off-Road Segmentation
Adapts pre-trained SegFormer to custom off-road datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers import TrainingArguments, Trainer
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OffRoadDataset(Dataset):
    """
    Custom dataset for off-road segmentation
    
    Expected folder structure:
    dataset/
        images/
            img001.jpg
            img002.jpg
            ...
        masks/
            img001.png  (grayscale, pixel values = class IDs)
            img002.png
            ...
    """
    
    def __init__(self, 
                 images_dir, 
                 masks_dir, 
                 processor,
                 augment=True,
                 num_classes=150):
        """
        Args:
            images_dir: Path to images folder
            masks_dir: Path to masks folder
            processor: SegformerImageProcessor instance
            augment: Whether to apply data augmentation
            num_classes: Number of segmentation classes
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.processor = processor
        self.num_classes = num_classes
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + 
                                 list(self.images_dir.glob("*.png")))
        
        logger.info(f"Found {len(self.image_files)} images in {images_dir}")
        
        # Data augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.masks_dir / img_path.name.replace('.jpg', '.png')
        if not mask_path.exists():
            mask_path = self.masks_dir / img_path.name
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Process with SegformerImageProcessor
        encoded_inputs = self.processor(image, return_tensors="pt")
        
        # Remove batch dimension
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze()
        
        # Resize mask to match model output size
        mask = cv2.resize(mask, (encoded_inputs['pixel_values'].shape[-1], 
                                 encoded_inputs['pixel_values'].shape[-2]),
                         interpolation=cv2.INTER_NEAREST)
        
        encoded_inputs["labels"] = torch.from_numpy(mask).long()
        
        return encoded_inputs


def compute_metrics(eval_pred):
    """Compute IoU and accuracy metrics"""
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)
    
    # Upsample logits to match label size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    
    predicted = upsampled_logits.argmax(dim=1)
    
    # Compute metrics
    mask = labels != 255  # Ignore index
    correct = (predicted == labels) & mask
    
    accuracy = correct.sum().item() / mask.sum().item()
    
    # Compute mean IoU
    num_classes = logits.shape[1]
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_cls = predicted == cls
        label_cls = labels == cls
        
        intersection = (pred_cls & label_cls & mask).sum().item()
        union = (pred_cls | label_cls) & mask
        union = union.sum().item()
        
        if union > 0:
            iou_per_class.append(intersection / union)
    
    mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0
    
    return {
        "accuracy": accuracy,
        "mean_iou": mean_iou,
    }


def train_model(
    train_images_dir,
    train_masks_dir,
    val_images_dir,
    val_masks_dir,
    output_dir="./fine_tuned_model",
    base_model="nvidia/segformer-b0-finetuned-ade-512-512",
    num_epochs=20,
    batch_size=4,
    learning_rate=5e-5,
    num_classes=150,
    use_fp16=True
):
    """
    Fine-tune SegFormer on custom dataset
    
    Args:
        train_images_dir: Training images directory
        train_masks_dir: Training masks directory
        val_images_dir: Validation images directory
        val_masks_dir: Validation masks directory
        output_dir: Where to save fine-tuned model
        base_model: Base model to fine-tune from
        num_epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        learning_rate: Learning rate
        num_classes: Number of segmentation classes
        use_fp16: Use mixed precision training
    """
    
    logger.info("=" * 60)
    logger.info("FINE-TUNING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Base Model: {base_model}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"FP16: {use_fp16}")
    logger.info("=" * 60)
    
    # Load processor and model
    processor = SegformerImageProcessor.from_pretrained(base_model)
    model = SegformerForSemanticSegmentation.from_pretrained(
        base_model,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Create datasets
    train_dataset = OffRoadDataset(
        train_images_dir, 
        train_masks_dir, 
        processor, 
        augment=True,
        num_classes=num_classes
    )
    
    val_dataset = OffRoadDataset(
        val_images_dir, 
        val_masks_dir, 
        processor, 
        augment=False,
        num_classes=num_classes
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        fp16=use_fp16 and torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate
    logger.info("Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final Validation Accuracy: {eval_metrics['eval_accuracy']:.4f}")
    logger.info(f"Final Validation mIoU: {eval_metrics['eval_mean_iou']:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)
    
    return trainer, eval_metrics


def create_sample_dataset_structure(base_dir="./sample_dataset"):
    """
    Create sample dataset folder structure
    """
    base_path = Path(base_dir)
    
    # Create directories
    (base_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (base_path / "train" / "masks").mkdir(parents=True, exist_ok=True)
    (base_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (base_path / "val" / "masks").mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_content = """
# Off-Road Segmentation Dataset

## Structure
```
sample_dataset/
├── train/
│   ├── images/     # Training images (.jpg or .png)
│   └── masks/      # Training masks (.png, grayscale)
└── val/
    ├── images/     # Validation images
    └── masks/      # Validation masks
```

## Mask Format
- Grayscale PNG images
- Pixel values represent class IDs (0-149 for ADE20K classes)
- Use 255 for ignore/unlabeled pixels

## Class Mapping (Example for Off-Road)
- 0: Background/Unknown
- 6: Road/Path (safe)
- 9: Grass (safe)
- 13: Earth/Dirt (safe)
- 21: Water (unsafe)
- 12: Person (unsafe)
- 20: Vehicle (unsafe)

## Recommended Dataset Size
- Minimum: 100 training images + 20 validation images
- Good: 500+ training images + 100+ validation images
- Excellent: 2000+ training images + 500+ validation images

## Data Collection Tips
1. Extract frames from off-road videos
2. Use labelme or CVAT for annotation
3. Ensure diverse conditions (lighting, weather, terrain)
4. Balance safe/unsafe terrain examples
"""
    
    with open(base_path / "README.txt", "w") as f:
        f.write(readme_content)
    
    logger.info(f"Sample dataset structure created at: {base_dir}")
    logger.info("Add your images and masks to the appropriate folders")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer for Off-Road Segmentation")
    parser.add_argument("--train-images", type=str, required=True, help="Training images directory")
    parser.add_argument("--train-masks", type=str, required=True, help="Training masks directory")
    parser.add_argument("--val-images", type=str, required=True, help="Validation images directory")
    parser.add_argument("--val-masks", type=str, required=True, help="Validation masks directory")
    parser.add_argument("--output", type=str, default="./fine_tuned_model", help="Output directory")
    parser.add_argument("--base-model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-classes", type=int, default=150, help="Number of classes")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 training")
    parser.add_argument("--create-structure", action="store_true", help="Create sample dataset structure")
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_sample_dataset_structure()
    else:
        train_model(
            train_images_dir=args.train_images,
            train_masks_dir=args.train_masks,
            val_images_dir=args.val_images,
            val_masks_dir=args.val_masks,
            output_dir=args.output,
            base_model=args.base_model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_classes=args.num_classes,
            use_fp16=not args.no_fp16
        )
