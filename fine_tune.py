import os
import torch
from datasets import Dataset, DatasetDict, Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, TrainingArguments, Trainer
from torchvision.transforms import ColorJitter
from PIL import Image as PILImage
import numpy as np
import evaluate

# Configuration
MODEL_NAME = "nvidia/mit-b0" # Start with a smaller encoder for fine-tuning
OUTPUT_DIR = "./segformer_finetuned"
num_epochs = 10
batch_size = 4
learning_rate = 0.00006

def get_dataset(root_dir):
    """
    Expects directory structure:
    root_dir/
      train/
        images/
        masks/
      val/
        images/
        masks/
    """
    image_paths_train = sorted([os.path.join(root_dir, "train/images", x) for x in os.listdir(os.path.join(root_dir, "train/images"))])
    mask_paths_train = sorted([os.path.join(root_dir, "train/masks", x) for x in os.listdir(os.path.join(root_dir, "train/masks"))])
    image_paths_val = sorted([os.path.join(root_dir, "val/images", x) for x in os.listdir(os.path.join(root_dir, "val/images"))])
    mask_paths_val = sorted([os.path.join(root_dir, "val/masks", x) for x in os.listdir(os.path.join(root_dir, "val/masks"))])

    train_dataset = Dataset.from_dict({"image": image_paths_train, "label": mask_paths_train})
    val_dataset = Dataset.from_dict({"image": image_paths_val, "label": mask_paths_val})

    train_dataset = train_dataset.cast_column("image", Image())
    train_dataset = train_dataset.cast_column("label", Image())
    val_dataset = val_dataset.cast_column("image", Image())
    val_dataset = val_dataset.cast_column("label", Image())

    return DatasetDict({"train": train_dataset, "test": val_dataset})

# Metrics
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=2, # CHANGE THIS TO YOUR NUM CLASSES
            ignore_index=255,
            reduce_labels=False,
        )
        return {
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"],
        }

def main():
    # 1. Load Data
    # dataset = get_dataset("./data/my_dataset") # Uncomment and set path
    print("This script is a template. Uncomment the dataset loading lines to use.")
    return 

    # 2. Preprocessing
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)

    def train_transforms(example_batch):
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["label"]]
        inputs = processor(images, labels, return_tensors="pt")
        return inputs

    def val_transforms(example_batch):
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["label"]]
        inputs = processor(images, labels, return_tensors="pt")
        return inputs

    # train_ds.set_transform(train_transforms)
    # val_ds.set_transform(val_transforms)

    # 3. Model
    id2label = {0: "background", 1: "road"} # UPDATE THIS
    label2id = {v: k for k, v in id2label.items()}
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
    )

    # 4. Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        logging_steps=20,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()
