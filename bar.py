import pyaudio as pa
import matplotlib.pyplot as plt
import scipy.fftpack as fourier
import numpy as np
from scipy import signal
import struct
a = pa.PyAudio()

FRAMES = 1024*6
FORMATO = pa.paInt16
CANAL = 1
Fs = 44100

stream = a.open(
    format = FORMATO,
    channels =  CANAL,
    rate = Fs,
    input = True,
    output = True,
    frames_per_buffer = FRAMES
    )
# Creamos nuestra grafica y configurar los ejes
fig ,ax= plt.subplots(1)
aud_x = np.arange(0,FRAMES,1)
x_fft = np.linspace(0,Fs,FRAMES) #

line, = ax.plot(aud_x,np.zeros(FRAMES))
line_fft, = ax.semilogx(x_fft,np.zeros(FRAMES)) #Define la escala para el ejex para la señal en frecuencia
ax.set_ylim(-10,10) #ejey  
ax.set_xlim(1,5000) #ejex
fig.show()

while True:
    data = stream.read(FRAMES)
    dataInt = struct.unpack(str(FRAMES)+ "h",data) #dataInt es la señal del microfono en el tiempo
    
    yx = np.asarray(dataInt) #dataInt inicialmente es tupla y se trabajara cono array numpy
    #Normalizacion de la señal yx a +-1
    ymax = np.max(yx)
    ymin = np.min(yx)
    yx = 2*(yx - ymin)/(ymax -ymin)-1 #yx ya normalizado
    #Sacamos la transformada y luego la magnitud -> M
    M = abs(fourier.fft(yx))
    #Deinifimos un eje y dinamico segun la amplitud de la Magnitud ->M
    ax.set_ylim(0,np.max(M)+10)
    #Graficamos M
    line_fft.set_ydata(M)
    fig.canvas.draw()
    fig.canvas.flush_events()











