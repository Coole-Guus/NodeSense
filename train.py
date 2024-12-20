import os
import json
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Audio, ClassLabel, load_from_disk
from transformers import (
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import itertools
import soundata
import numpy as np

## Required for compatibility with the soundata library
np.float_ = np.float64

# Download and validate the UrbanSound8K dataset.
dataset = soundata.initialize('urbansound8k')
dataset.download()
dataset.validate()

DATA_DIR = "urbansound8k" # HARDCODED PATH, this needs to be changed to your own path where the UrbanSound8K dataset is stored.
CSV_PATH = os.path.join(DATA_DIR, "metadata/UrbanSound8K.csv")
SAMPLE_RATE = 16000
MAX_LENGTH = 64000  # 64000 = 4 seconds at the 16Khz sample rate
EPOCHS = 5          
LEARNING_RATE = 1e-4

# Read in metadata.
metadata = pd.read_csv(CSV_PATH)

# Distribute folds over a training set, validation set and test set.
train_folds = [f for f in range(1, 9)]
val_folds = [9]
test_folds = [10]

# Create a dataset from the folds.
def make_dataset_from_folds(fold_list):
    subset = metadata[metadata.fold.isin(fold_list)]
    files = []
    labels = []
    for _, row in subset.iterrows():
        path = os.path.join(DATA_DIR, "audio", f"fold{row.fold}", row.slice_file_name)
        files.append(path)
        labels.append(row.classID)
    # Create a HuggingFace Dataset object and add audio and labels columns.
    ds = Dataset.from_dict({"audio": files, "label": labels})
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    ds = ds.cast_column(
        "label",
        ClassLabel(
            num_classes=len(metadata.classID.unique()),
            names=sorted(metadata["class"].unique())
        )
    )
    return ds

# Create the training, validation and testing datasets.
train_dataset = make_dataset_from_folds(train_folds)
val_dataset = make_dataset_from_folds(val_folds)
test_dataset = make_dataset_from_folds(test_folds)

# Create the final Dataset Dictionary HuggingFace object.
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset,
})

# As a check, print number of samples in all datasets.
print("Number of training samples:", len(dataset["train"]))
print("Number of validation samples:", len(dataset["validation"]))
print("Number of test samples:", len(dataset["test"]))

# We don't really need a tokenizer since this is an classification task
# However, the model does not have a standard tokenizer, and for it to work
# we need a sample tokenizer. 
vocab = {
    "|": 0,     # CTC blank
    "a": 1,     # dummy character
    "b": 2      # dummy character
}

# Open vocab.json, write the vocab to it.
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

# Load tokenizer, feature extrctor and processor.
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Pad all audio files with silence to 4 seconds (if they are shorter).
def pad_with_silence(audio, target_length):
    length = len(audio)
    if length > target_length:
        return audio[:target_length]
    elif length < target_length:
        padding = np.zeros(target_length - length, dtype=audio.dtype)
        return np.concatenate([audio, padding])
    return audio

# Preprocess function for the dataset.
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    adjusted_audio = [pad_with_silence(a, MAX_LENGTH) for a in audio_arrays]

    inputs = processor(
        adjusted_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="longest"
    )
    inputs["labels"] = examples["label"]
    return inputs

## First run: Map and save, otherwise, use the one loaded from disk.
processed_dataset = dataset.map(preprocess_function, batch_size=16, batched=True)
processed_dataset.save_to_disk("Saved_Processed_Dataset")

# When saved in a previous run, comment the two lines above and uncomment the line below!
# processed_dataset = load_from_disk("Saved_Processed_Dataset")

label_list = processed_dataset["train"].features["label"].names
num_labels = len(label_list)

# Load the model.
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/hubert-base-ls960",
    num_labels=num_labels,
    problem_type="single_label_classification"
)

# Load the metrics (accuracy, precision, recall).
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# This function computes the metrics and returns them as dictionary back to the trainer.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    accuracy = accuracy_score(labels, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Initialize values of hyperparameters to use in grid search.
learning_rates = [1e-5, 2e-5, 5e-5]
batch_sizes = [8, 16]
weight_decays = [0, 0.01]

# Initialize gridsearch.
gridSearch = list(itertools.product(learning_rates, batch_sizes, weight_decays))

# Initialize variables to keep track of best combination of values of hyperparameters
best_accuracy = 0
best_hyperparams = {}

# For every combination in the grid, train the model and evaluate on the validation set.
for learning_rate, batch_size, weight_decay in gridSearch:
    # Print current values.
    print(f"Training with lr={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay}")

    # Update training arguments.
    training_args = TrainingArguments(
        output_dir="./urban_sound_checkpoints",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=5,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none"
    )

    # Initialize the trainer.
    trainer = Trainer(
        model=model,  # Use the initialized model.
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=processor,
        compute_metrics=compute_metrics
    )

    # Train and evaluate.
    trainer.train()
    eval_results = trainer.evaluate(processed_dataset["validation"])
    print(f"Validation results: {eval_results}")

    # Check if this is the best model.
    if eval_results["eval_accuracy"] > best_accuracy:
        best_accuracy = eval_results["eval_accuracy"]
        best_hyperparams = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay
        }

    # Print the best hyperparameters.
    print(f"Best hyperparameters: {best_hyperparams} with accuracy {best_accuracy}")

# Evaluate on test set.
best_training_args = TrainingArguments(
    output_dir="./urban_sound_checkpoints",
    per_device_train_batch_size=best_hyperparams["batch_size"],
    per_device_eval_batch_size=best_hyperparams["batch_size"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=5,
    learning_rate=best_hyperparams["learning_rate"],
    weight_decay=best_hyperparams["weight_decay"],
    gradient_accumulation_steps=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="none"
)

# Initialize trainer with best hyperparameter values.
trainer = Trainer(
    model=model,
    args=best_training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# Train the model with best hyperparameter values.
trainer.train()

# Evaluate on the test set.
test_results = trainer.evaluate(processed_dataset["test"])
print(f"Test results: {test_results}")

# Save the fine-tuned model and processor with best hyperparameter values.
trainer.save_model("./urban_sound_model")
processor.save_pretrained("./urban_sound_model")

