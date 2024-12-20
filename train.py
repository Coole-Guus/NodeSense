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

# Hardcoded path (please change for yourself)
DATA_DIR = "/path_to_urbansound8k"
CSV_PATH = os.path.join(DATA_DIR, "metadata/UrbanSound8K.csv")
SAMPLE_RATE = 16000
MAX_LENGTH = 64000  # 64000 = 4 seconds at the 16Khz sample rate
EPOCHS = 5          
LEARNING_RATE = 1e-4

metadata = pd.read_csv(CSV_PATH)

train_folds = [f for f in range(1, 9)]
val_folds = [9]
test_folds = [10]

def make_dataset_from_folds(fold_list):
    subset = metadata[metadata.fold.isin(fold_list)]
    files = []
    labels = []
    for _, row in subset.iterrows():
        path = os.path.join(DATA_DIR, "audio", f"fold{row.fold}", row.slice_file_name)
        files.append(path)
        labels.append(row.classID)
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

train_dataset = make_dataset_from_folds(train_folds)
val_dataset = make_dataset_from_folds(val_folds)
test_dataset = make_dataset_from_folds(test_folds)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset,
})

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

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def pad_with_silence(audio, target_length):
    length = len(audio)
    if length > target_length:
        return audio[:target_length]
    elif length < target_length:
        padding = np.zeros(target_length - length, dtype=audio.dtype)
        return np.concatenate([audio, padding])
    return audio

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

## First run: Map and save, otherwise, use the one loaded from disk
# processed_dataset = dataset.map(preprocess_function, batch_size=16, batched=True)
# processed_dataset.save_to_disk("testje")
processed_dataset = load_from_disk("testje")
label_list = processed_dataset["train"].features["label"].names
num_labels = len(label_list)


model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/hubert-base-ls960",
    num_labels=num_labels,
    problem_type="single_label_classification"
)

# Load metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

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

# Fine-tune model on the Urbansound8K dataset
training_args = TrainingArguments(
    output_dir="./urban_sound_checkpoints",
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor,
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(processed_dataset["test"])
print("Test Results:", test_results)

trainer.save_model("./urban_sound_model")
processor.save_pretrained("./urban_sound_model")

