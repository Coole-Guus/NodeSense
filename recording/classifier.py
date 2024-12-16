from io import BytesIO

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification


class Classifier:

    def __init__(self):
        self.labels = {
            "LABEL_0": "Air conditioner",
            "LABEL_1": "Car horn",
            "LABEL_2": "Children playing",
            "LABEL_3": "Dog barking",
            "LABEL_4": "Drilling",
            "LABEL_5": "Enginge idling",
            "LABEL_6": "Gun shot",
            "LABEL_7": "Jackhammer",
            "LABEL_8": "Siren",
            "LABEL_9": "Street music"
        }
        self.processor = Wav2Vec2Processor.from_pretrained("../urban_sound_model")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained("../urban_sound_model")
        self.model.eval()

    def classify(self, data: bytes, sample_rate: int):
        audio_tensor = torch.frombuffer(data, dtype=torch.int16)

        # Normalize to [-1, 1]. (-)2^15 is max int16 value
        audio_tensor = audio_tensor.float() / (2 ** (16 - 1))

        # Resample if SR != 16000
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_array = audio_tensor.squeeze().numpy()
        inputs = self.processor(audio_array, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get predicted class ID
        pred_ids = torch.argmax(logits, dim=-1)
        pred_id = pred_ids[0].item()
        predicted_label = self.model.config.id2label[pred_id]
        print("Predicted class:", self.labels[predicted_label])
