import os
import json
import pandas as pd
import numpy as np
import torch
import gc
from datasets import Dataset, DatasetDict, Audio, ClassLabel
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
import soundata

# Download and validate the UrbanSound8K dataset.
dataset = soundata.initialize('urbansound8k')
dataset.download()
dataset.validate()

DATA_DIR = "urbansound8k"  # HARDCODED PATH, this needs to be changed to your own path where the UrbanSound8K dataset is stored.
CSV_PATH = os.path.join(DATA_DIR, "metadata/UrbanSound8K.csv")
SAMPLE_RATE = 16000
MAX_LENGTH = 64000
EPOCHS = 3  
LEARNING_RATE = 1e-4

# Read in metadata.
metadata = pd.read_csv(CSV_PATH)

# Folds are from 1 to 10, save them all in one list.
all_folds = list(range(1, 11))

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

# We will run a loop for cross-validation
# For each fold i: use fold i as test, fold (i%10)+1 as val, rest as train
all_test_results = []

for i in range(1, 11):
    test_fold = i
    val_fold = (i % 10) + 1
    train_folds = [f for f in all_folds if f not in [test_fold, val_fold]]
    
    # Create datasets for this iteration
    train_dataset = make_dataset_from_folds(train_folds)
    val_dataset = make_dataset_from_folds([val_fold])
    test_dataset = make_dataset_from_folds([test_fold])
    
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })
    
    print(f"\n--- Cross-Validation Iteration {i} ---")
    print("Train folds:", train_folds)
    print("Validation fold:", val_fold)
    print("Test fold:", test_fold)
    print("Number of training samples:", len(dataset["train"]))
    print("Number of validation samples:", len(dataset["validation"]))
    print("Number of test samples:", len(dataset["test"]))
    
    # Preprocess the dataset using the preprocess_function(...).
    processed_dataset = dataset.map(preprocess_function, batch_size=16, batched=True)
    label_list = processed_dataset["train"].features["label"].names
    num_labels = len(label_list)

    # Load the model.
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/hubert-base-ls960",
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

    # Initialize training arguments.
    training_args = TrainingArguments(
        output_dir=f"./urban_sound_checkpoints_fold_{i}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none" 
    )

    # Initialize the trainer.
    trainer = Trainer(
        model=model, # Use the initialized model.
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=processor,
        compute_metrics=compute_metrics
    )

    # Train for this fold.
    trainer.train()

    # Evaluate on test set for this fold.
    test_results = trainer.evaluate(processed_dataset["test"])
    print("Test Results for fold", test_fold, ":", test_results)
    all_test_results.append(test_results)

    # Save the model and processor for this fold.
    trainer.save_model(f"./urban_sound_model_fold_{i}")
    processor.save_pretrained(f"./urban_sound_model_fold_{i}")

    # Clear memory (due to the need to a lot of RAM).
    del model
    del trainer
    del processed_dataset
    del dataset
    torch.cuda.empty_cache()
    gc.collect()

# After all folds are done, average the test results.
accs = [r["eval_accuracy"] for r in all_test_results]
precisions = [r["eval_precision"] for r in all_test_results]
recalls = [r["eval_recall"] for r in all_test_results]
f1s = [r["eval_f1"] for r in all_test_results]

# Print all average results to the user.
print("\n--- Cross-Validation Summary ---")
print(f"Average Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Average F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
