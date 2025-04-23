import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------ Data Preprocessing ------------------------

class EmailClassification(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """Initialize dataset with texts, labels and tokenizer settings."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of sample in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_class_weights(labels):
    """Compute class weights inversely proportional to class frequencies."""
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float)


# ------------------------ Model Setup ------------------------

class WeightedTrainer(Trainer):
    """Custom trainer that uses weighted loss function for imbalanced data."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=0.1
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss using the weighted loss function."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def load_model_and_tokenizer(model_name, label2id, id2label):
    """Load the model and tokenizer."""
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer

# ------------------------ Metrics ------------------------

def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    report = classification_report(
        labels,
        preds,
        target_names=list(label2id.keys()),
        output_dict=True
    )

    results = {
        'accuracy': report['accuracy'],
        'f1_macro': report['macro avg']['f1-score'],
        'f1_weighted': report['weighted avg']['f1-score'],
        'precision_weighted': report['weighted avg']['precision'],
        'recall_weighted': report['weighted avg']['recall']
    }

    for cls_name, cls_id in label2id.items():
        results[f'f1_{cls_name}'] = report[cls_name]['f1-score']

    return results


# ------------------------ Training and Evaluation ------------------------

def train_model(df, model_name, label2id, id2label, max_length, batch_size, epochs, seed):
    """Train and evaluate model using train/val/test split."""
    df['label_id'] = df['type'].map(label2id)

    # Split the data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['email_processed'].values,
        df['label_id'].values,
        test_size=0.3,
        stratify=df['label_id'].values,
        random_state=seed
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed
    )

    # Create datasets
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    train_dataset = EmailClassification(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmailClassification(val_texts, val_labels, tokenizer, max_length)
    test_dataset = EmailClassification(test_texts, test_labels, tokenizer, max_length)

    # Compute class weights
    class_weights = compute_class_weights(train_labels)

    # Load the model
    model, tokenizer = load_model_and_tokenizer(model_name, label2id, id2label)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=120,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        fp16=True,
        gradient_accumulation_steps=4
    )

    # Trainer with Weighted Loss
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights
    )

    # Train the model
    trainer.train()

    # Evaluate on validation and test set
    val_results = trainer.evaluate()
    test_results = trainer.predict(test_dataset)
    test_metrics = compute_metrics(test_results)

    # Print results
    print("\nValidation Results:")
    for metric_name, value in val_results.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nTest Results:")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Get predictions for test set
    test_preds = np.argmax(test_results.predictions, axis=1)

    print("\nTest Set Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=list(label2id.keys())))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label2id.keys()), yticklabels=list(label2id.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test_set.png')
    plt.close()

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')

    return model, tokenizer


# ------------------------ Main ------------------------

if __name__ == "__main__":
    label2id = {
        'Incident': 0,
        'Request': 1,
        'Problem': 2,
        'Change': 3
    }
    id2label = {v: k for k, v in label2id.items()}

    # Model and Training Configuration
    MODEL_NAME = "microsoft/mdeberta-v3-base"
    MAX_LENGTH = 128
    BATCH_SIZE = 4
    EPOCHS = 5
    SEED = 42

    # Load Dataset
    df = pd.read_csv("D:\OG Project\Data\combined_emails_with_natural_pii.csv")

    # Train and evaluate the model
    model, tokenizer = train_model(
        df=df,
        model_name=MODEL_NAME,
        label2id=label2id,
        id2label=id2label,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        seed=SEED
    )
