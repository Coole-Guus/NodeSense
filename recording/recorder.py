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
        self.min_recorded_chunks = 30

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
                peak_detection_value = rms

                peak_value = self.peak_detector.add_value(peak_detection_value)

                if peak_value == 1:
                    if not is_recording:
                        self.peak_detector.mark_start_recording()
                        print("Peak detected.")
                    is_recording = True
                    should_stop_recording = False
                elif peak_value == -1:
                    should_stop_recording = True

                if is_recording:
                    recorded_data += data

                recorded_samples = len(recorded_data) / (16 / 8)
                recorded_chunks = (recorded_samples / self.chunk_size)

                if should_stop_recording and is_recording and recorded_chunks >= self.min_recorded_chunks:
                    print("Peak ended.")
                    self.peak_detector.mark_end_recording()
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
