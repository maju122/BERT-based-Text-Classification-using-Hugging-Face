"""
BERT-based Text Classification using Hugging Face Transformers
Dataset : IMDb (binary sentiment classification)
Model   : bert-base-uncased (fine-tuned)

Run in Google Colab (GPU runtime) or locally with a CUDA GPU.
Install dependencies first:
    pip install transformers datasets torch scikit-learn matplotlib accelerate -U
"""

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
raw_datasets = load_dataset("stanfordnlp/imdb")

# Stratified subsample for faster lab-scale training (use full split for best results)
train_ds = raw_datasets["train"].shuffle(seed=42).select(range(5000))
test_ds = raw_datasets["test"].shuffle(seed=42).select(range(5000))

print(train_ds)
print(train_ds[0])

# ---------------------------------------------------------------------
# 2. Tokenization
# ---------------------------------------------------------------------
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )


train_tok = train_ds.map(tokenize_fn, batched=True)
test_tok = test_ds.map(tokenize_fn, batched=True)

train_tok = train_tok.rename_column("label", "labels")
test_tok = test_tok.rename_column("label", "labels")

train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------------------------------------------------------
# 3. Load pretrained BERT model with classification head
# ---------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


# ---------------------------------------------------------------------
# 4. Metrics function
# ---------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------
# 5. Training arguments and Trainer
# ---------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./bert-imdb",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=int(0.1 * (len(train_tok) / 16) * 3),  # ~10% of total training steps
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------------
# 6. Fine-tune
# ---------------------------------------------------------------------
train_result = trainer.train()

# ---------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------
eval_metrics = trainer.evaluate()
print("\nEvaluation metrics:")
print(eval_metrics)

# Detailed predictions for confusion matrix / classification report
predictions = trainer.predict(test_tok)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=-1)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix:")
print(cm)

# ---------------------------------------------------------------------
# 8. Plot confusion matrix
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
labels = ["Negative", "Positive"]
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticks([0, 1])
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix - IMDb Test Set")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 9. Plot training/validation loss and accuracy curves
# ---------------------------------------------------------------------
log_history = trainer.state.log_history
epochs, train_losses, eval_losses, eval_accs = [], [], [], []
for entry in log_history:
    if "loss" in entry and "epoch" in entry:
        train_losses.append((entry["epoch"], entry["loss"]))
    if "eval_loss" in entry:
        eval_losses.append((entry["epoch"], entry["eval_loss"]))
        eval_accs.append((entry["epoch"], entry.get("eval_accuracy")))

if eval_losses:
    ep, el = zip(*eval_losses)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, el, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()

# ---------------------------------------------------------------------
# 10. Try the classifier on custom sentences
# ---------------------------------------------------------------------
import torch

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sample_reviews = [
    "An absolute masterpiece - the acting, the score, the pacing, everything works.",
    "I wanted to like this movie but the plot was all over the place.",
]

inputs = tokenizer(sample_reviews, padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

for text, pred, prob in zip(sample_reviews, preds, probs):
    label = "Positive" if pred.item() == 1 else "Negative"
    confidence = prob[pred].item()
    print(f"\nReview: {text}\n -> Predicted: {label} (confidence: {confidence:.3f})")

# ---------------------------------------------------------------------
# 11. Save the fine-tuned model
# ---------------------------------------------------------------------
trainer.save_model("./bert-imdb-final")
tokenizer.save_pretrained("./bert-imdb-final")
print("\nModel saved to ./bert-imdb-final")
