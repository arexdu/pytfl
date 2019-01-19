import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

#from scipy.io.wavefile import write

spf = wave.open('./test.wav', 'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
print("numpy signal shape:",signal.shape)

plt.plot(signal)
plt.title("test wav without echo")
plt.show()
