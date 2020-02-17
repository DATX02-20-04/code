import pyaudio
import wave
import numpy as np
import random as rng


vol = 0.1
sr = 44100  # in hz
dur = 1

f = rng.random() * 20000

samples = np.sin(2*np.pi*np.arange(sr*dur)*f/sr).astype(np.float32)

pa = pyaudio.PyAudio()

audioStream = pa.open(
    format=pyaudio.paFloat32,
    channels=1,
    output=True,
    rate=sr
)

audioStream.write((vol * samples).tobytes())

audioStream.stop_stream()
audioStream.close()

pa.terminate()


wf = wave.open("{htz}.wav".format(htz=str(f)), 'wb')
wf.setnchannels(1)
wf.setsampwidth(pa.get_sample_size(pyaudio.paFloat32))
wf.setframerate(sr)
wf.writeframes((vol*samples).tobytes())

wf.close()
