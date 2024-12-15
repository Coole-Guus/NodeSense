from math import log10

import pyaudio
import audioop

from peak_detector import PeakDetector

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

audio = pyaudio.PyAudio()

print(audio.get_default_input_device_info())

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    frames_per_buffer=CHUNK,
    input=True,
)

print("Starting recording. Press CTRL+C to stop.")

peak_detector = PeakDetector([], lag=30, threshold=5, influence=0)

try:
    while True:
        data = stream.read(CHUNK)
        rms = max(audioop.rms(data, 1), 1)
        db = 20 * log10(rms)

        peak_detector.add_value(rms)
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    peak_detector.plot()
    print("Stopped recording.")
