import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import audioop

import torch

from peak_detector import PeakDetector

from classifier import Classifier


class Recorder:
    def __init__(self):
        self.chunk_size = 2048
        self.format = pyaudio.paInt16
        self.num_channels = 1
        self.sample_rate = 44100
        self.min_recorded_chunks = 30  # roughly 1.5 seconds (30 * 2048 = 122880 / 44100 = ~3)

        self.recorder = pyaudio.PyAudio()
        self.peak_detector = PeakDetector([], lag=10, threshold=5, influence=1)
        self.classifier = Classifier()

        self.stream = None

    def start(self):
        if self.stream is None:
            self.stream = self.recorder.open(
                format=self.format,
                channels=self.num_channels,
                rate=self.sample_rate,
                frames_per_buffer=self.chunk_size,
                input=True,
            )
            print("Started recording. Press CTRL+C to stop.")

        try:
            is_recording = False
            should_stop_recording = False
            recorded_data = b''

            while True:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                rms = audioop.rms(data, self.num_channels)
                # amp_peak = np.max(np.abs(np.frombuffer(data, dtype=np.int16)))

                peak_detection_value = rms
                # peak_detection_value = amp_peak

                # Prints root-mean-square, decibel and absolute peak
                # print(rms, 20 * np.log10(rms / 32768) if rms > 0 else -np.inf, amp_peak)

                out = self.peak_detector.add_value(peak_detection_value)

                if out == 1:
                    if not is_recording:
                        print("Peak detected.")
                    is_recording = True
                    should_stop_recording = False
                elif out == -1:
                    should_stop_recording = True

                if is_recording:
                    recorded_data += data

                recorded_samples = len(recorded_data) / (16 / 8)
                recorded_chunks = recorded_samples / self.chunk_size

                if should_stop_recording and is_recording and recorded_chunks >= self.min_recorded_chunks:
                    print("Peak ended.")
                    self.classifier.classify(recorded_data, self.sample_rate)
                    recorded_data = b''
                    is_recording = False
                    should_stop_recording = False

        except KeyboardInterrupt:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self.recorder.terminate()
            self.peak_detector.plot()
            print("Stopped recording.")