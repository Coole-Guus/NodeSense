#%% md
# # Preprocessing step
#%%
#%%
import soundata
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import ast
from transformers import ASTForAudioClassification, AdamW, get_scheduler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  # Import tqdm

## Required for compatibility with the soundata library
np.float_ = np.float64

dataset = soundata.initialize('urbansound8k')
dataset.download()
dataset.validate()

example_clip = dataset.choice_clip()
print(example_clip)
#%% md
# ## melspectogram instructions
# Goal: mel spectogram with resolution of 48 frequencies from 40hz upto 16khz.
# with 32 time steps of 1/16th of a second with a half overlap during 1 second.
# We assume the data is streaming.
# data from the last 1/32th of the last second of the previous sample is added to the next.
# so in total 48 frequency bands and 32 spectra
# - audio sample between 1 and 16 seconds
# - apply hann window function
# - apply fourier transform to get absolute spectrum
# - summize spectral components using MEL filters.
# MEL spectral filters should be done using melbankm.m from VOICEBOX speech processing toolbox. http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
# as follows: [m,mc,mn]=melbankm(p,n,fs,fl,fh,w) where p=48,n=3000,fs=48000,fl=40,fh=16000,w="ch"
# output m is matrix of filter values, mc and mn are frequency border and indices
# - After this output the matrix is edited as follows
#   - log on all components using natural log and a low pass filter of epsil = 1e-6, so x = log(Svalue, epsil)
#   - normalize the mel spectro gram for training and classification, so all melspectograms have max value of 2:
# Xnorm(:,:) = X(:,:)- max(max(X(:,:))) +2;
#%%


sample_rate = 48000
number_fft = 3000
number_mel_bands = 48
frequency_min = 40
frequency_max = 16000
#%%
mel_filter_bank = librosa.filters.mel(sr=sample_rate,
                                      n_fft=number_fft,
                                      n_mels=number_mel_bands,
                                      fmin=frequency_min,
                                      fmax=frequency_max,
                                      htk=True)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_filter_bank, x_axis='linear')
plt.colorbar()
plt.title('Mel filter bank')
plt.tight_layout()
plt.show()
#%%
half_overlap = number_fft // 2

def readAudioSample(clip):
    audio, _ = librosa.load(clip.audio_path, sr=sample_rate)
    return audio

def applyHannWindow(audio):
    audio_windowed = audio * np.hanning(len(audio))
    return audio_windowed

def computeSpectrogram(audio_windowed):
    S = np.abs(librosa.stft(audio_windowed, n_fft=number_fft, hop_length=half_overlap))
    return S

def applyMelFilterBank(S):
    S_mel = np.dot(mel_filter_bank, S)
    return S_mel

def processMelSpectrogram(S_mel):
    epsil = 1e-6
    S_mel_log = np.log(S_mel + epsil)
    S_mel_norm = S_mel_log - np.max(S_mel_log) + 2
    return S_mel_norm
#%%
def plotSpectrogram(S, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             sr=sample_rate,
                             fmin=frequency_min,
                             fmax=frequency_max,
                             hop_length=half_overlap,
                             x_axis='time',
                             y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
#%%
clip = dataset.choice_clip()

audio = readAudioSample(clip)
audio_windowed = applyHannWindow(audio)
S = computeSpectrogram(audio_windowed)
S_mel = applyMelFilterBank(S)
S_mel_processed = processMelSpectrogram(S_mel)
print(clip.tags.labels)
plotSpectrogram(S_mel_processed, 'Processed Mel spectrogram')

def validate_mel_spectrogram(S_mel_processed):
    if not (S_mel_processed.min() >= np.log(1e-6) and S_mel_processed.max() <= 2):
        raise ValueError("Frequency range is not valid")
    return True

# Validate the processed mel spectrogram
is_valid = validate_mel_spectrogram(S_mel_processed)
print(f"Validation result: {is_valid}")
#%% md
# # transforming the data
#%%
def getSpectrogram(clip):
    audio = readAudioSample(clip)
    audio_windowed = applyHannWindow(audio)
    S = computeSpectrogram(audio_windowed)
    S_mel = applyMelFilterBank(S)
    S_mel_processed = processMelSpectrogram(S_mel)
    return S_mel_processed
#%%
import pandas as pd

data = []

for key, clip in dataset.load_clips().items():
    spec = getSpectrogram(clip)
    data.append([key, clip.audio_path, spec.tolist()])

df = pd.DataFrame(data, columns=['Key', 'AudioPath', 'Spectrogram'])

df.to_csv("data.csv", index=False)
#%% md
# # below is debugging
#%%
import numpy as np

# Random sample
audio_sample_duration = np.random.randint(1, 17)
total_samples = int(sample_rate * audio_sample_duration)

audio_data = np.random.randn(total_samples)
audio_windowed = applyHannWindow(audio_data)

S = computeSpectrogram(audio_windowed)
S_mel = applyMelFilterBank(S)
print(S_mel)

metadata = pd.read_csv("urbansound8k/metadata/UrbanSound8K.csv")
allData = pd.read_csv("data.csv")

print(metadata.columns)  # Verify key column in metadata
print(allData.columns)   # Verify key column in allData
allData['Key'] = allData['Key'].astype(str)
metadata['slice_file_name'] = metadata['slice_file_name'].astype(str)
metadata['slice_file_name'] = metadata['slice_file_name'].str.replace('.wav', '', regex=False)

trainMetadata = metadata[metadata['fold'] < 9]
valMetadata = metadata[metadata['fold'] == 9]
testMetadata = metadata[metadata['fold'] == 10]

# Define the target length for your spectrograms (e.g., 129)
target_length = 129


# Function to repeat (loop) spectrograms to match target length
def repeat_spectrogram(spectrogram, target_length):
    for bin in range(len(spectrogram)):
        # Calculate how many times to repeat the spectrogram to reach or exceed the target length
        repeat_times = (target_length // len(spectrogram[bin])) + 1  # +1 ensures we have enough data

        # Repeat the spectrogram and trim to the target length
        spectrogram[bin] = (spectrogram[bin] * repeat_times)[:target_length]

    return spectrogram

trainDF = pd.merge(allData[['Key', 'Spectrogram']], trainMetadata[['slice_file_name', 'class']], left_on='Key', right_on='slice_file_name', how='inner')
valDF = pd.merge(allData[['Key', 'Spectrogram']], valMetadata[['slice_file_name', 'class']], left_on='Key', right_on='slice_file_name', how='inner')
testDF = pd.merge(allData[['Key', 'Spectrogram']], testMetadata[['slice_file_name', 'class']], left_on='Key', right_on='slice_file_name', how='inner')

trainDF.drop(columns=['slice_file_name'], inplace=True)
valDF.drop(columns=['slice_file_name'], inplace=True)
testDF.drop(columns=['slice_file_name'], inplace=True)

trainDF['Spectrogram'] = trainDF['Spectrogram'].apply(ast.literal_eval)
valDF['Spectrogram'] = valDF['Spectrogram'].apply(ast.literal_eval)
testDF['Spectrogram'] = testDF['Spectrogram'].apply(ast.literal_eval)

print(len(trainDF['Spectrogram']))  # Print the first entry
print(len(trainDF['Spectrogram'][0]))  # Print the first entry
print(trainDF['Spectrogram'])
print(trainDF['Spectrogram'][0])
print(trainDF['Spectrogram'][0][0])

trainDF['Spectrogram'] = trainDF['Spectrogram'].apply(lambda x: repeat_spectrogram(x, target_length))
valDF['Spectrogram'] = valDF['Spectrogram'].apply(lambda x: repeat_spectrogram(x, target_length))
testDF['Spectrogram'] = testDF['Spectrogram'].apply(lambda x: repeat_spectrogram(x, target_length))

trainDF['Spectrogram Length'] = trainDF['Spectrogram'].apply(len)
print(trainDF['Spectrogram Length'].value_counts())  # This will show the distribution of lengths
print(len(trainDF['Spectrogram']))  # Print the first entry
print(len(trainDF['Spectrogram'][0]))  # Print the first entry
print(trainDF['Spectrogram'])
print(trainDF['Spectrogram'][0])
print(trainDF['Spectrogram'][0][0])

print(trainDF)
print(valDF)
print(testDF)

# Convert labels into integers using LabelEncoder
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(trainDF['class'])
val_labels = label_encoder.transform(valDF['class'])
test_labels = label_encoder.transform(testDF['class'])

train_spectrograms = torch.tensor(trainDF['Spectrogram'].to_list(), dtype=torch.float32)
val_spectrograms = torch.tensor(valDF['Spectrogram'].to_list(), dtype=torch.float32)
test_spectrograms = torch.tensor(testDF['Spectrogram'].to_list(), dtype=torch.float32)

# Create a custom Dataset class for PyTorch DataLoader
class AudioDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]

# Create DataLoaders for train, val, and test
train_dataset = AudioDataset(train_spectrograms, train_labels)
val_dataset = AudioDataset(val_spectrograms, val_labels)
test_dataset = AudioDataset(test_spectrograms, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print(train_loader)
print(val_loader)
print(test_loader)

from datasets import load_dataset

# ds = load_dataset("deetsadi/musiccaps_mel_spectrograms")
# model =
#
# # Load pre-trained AST model and feature extractor
# # model = ASTForAudioClassification.from_pretrained(
# #     "MIT/ast-finetuned-audioset-10-10-0.4593",
# #     num_labels=10,
# #     ignore_mismatched_sizes=True
# # )
#
# # model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"  # Adjust this to an AST model, like 'facebook/ast'
# # extractor = AutoFeatureExtractor.from_pretrained(model_name)
# # model = AutoModelForAudioClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
#
# # Move model to the appropriate device (GPU or CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # Example of training loop
# from torch.optim import Adam
# from torch.nn import CrossEntropyLoss
#
# optimizer = Adam(model.parameters(), lr=1e-5)
# criterion = CrossEntropyLoss()
#
# # Training loop
# for epoch in range(3):  # Number of epochs
#     model.train()
#     # Adding a progress bar to the training loop
#     train_loss = 0
#     with tqdm(train_loader, desc=f"Epoch {epoch+1}/{3}", unit="batch") as pbar:
#         for spectrograms, labels in pbar:
#             spectrograms = spectrograms.to(device)
#             labels = labels.to(device)
#
#             # Forward pass
#             outputs = model(spectrograms)
#             loss = criterion(outputs.logits, labels)
#
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # Update the progress bar description with loss info
#             pbar.set_postfix(loss=loss.item())
#
#             # Keep track of the total loss for this epoch
#             train_loss += loss.item()
#
#     # Evaluate on validation set after each epoch
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         with tqdm(val_loader, desc="Validation", unit="batch") as val_pbar:
#             for spectrograms, labels in val_pbar:
#                 spectrograms = spectrograms.to(device)
#                 labels = labels.to(device)
#
#                 outputs = model(spectrograms)
#                 loss = criterion(outputs.logits, labels)
#                 val_loss += loss.item()
#
#                 # Optionally update the progress bar
#                 val_pbar.set_postfix(val_loss=loss.item())
#
#         # Print validation loss after each epoch
#     print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")

# plotSpectrogram(S, 'Spectrogram')