"""
@author Noé Vásquez Godínez
@email noe-vg@outlook.com
@about
    Required modules
        - pyaudio
        - matplotlib
        - scipy
        - numpy
        - struct
"""
import random

# sys
import sys

# Threading
import threading

# Capture audio
import pyaudio as pyaudio

# Plot charts
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Library used for scientific computing
import scipy.fftpack as fourier
from scipy import signal

# Mathematical operations
import numpy as np

# Convert native Python data types
import struct

# Plot

globalDataStream = [[0],[0]]
globalDataFilter = [[0],[0]]

rows = 2
cols = 1



# Create an interface to PortAudio
audioRecord = pyaudio.PyAudio()

# Values to recording

# Record in chunks of 1024 times six samples
FRAMES = 1024*6
# Format bites per sample
FORMAT = pyaudio.paInt16
# Chanell to get
CHANNEL = 1
# Record at 44100 samples per second
SAMPLES = 44100
"""
Recording
"""
def recording(outData1,outData2):    
    # Get the stream
    stream = audioRecord.open(
        format = FORMAT,
        channels =  CHANNEL,
        rate = SAMPLES,
        input = True,
        output = True,
        frames_per_buffer = FRAMES
        )
    # Get bytes
    data = stream.read(FRAMES) 
    # Get time signal data
    outData = struct.unpack( str(FRAMES)+ "h",data)

    outData1[0] = range(0,len(outData))
    outData1[1] = outData

    outData2[0] = range(0,len(outData))
    outData2[1] = outData

"""
Plot data
"""
def plotData(num,ax1,ax2,data1,data2):
    ax1.plot( data1[0], data1[1],color='blue')
    ax2.plot( data2[0], data2[1],color='green')    
    return ax1,ax2
"""
This function implements the estimation of 
the heart rate by means of digital filters.
"""
def main():    
    # Cancel execute with a key input with ctrl + c
    try:
        while True:            
            fig, (ax1,ax2) = plt.subplots(2)
            fig.suptitle('Filtros FIR')

            ax1.set_title('Real stream')
            ax1.plot( globalDataStream[0], globalDataStream[1] )
            
            ax2.set_title('Filter FIR')
            ax2.plot( globalDataStream[0], globalDataFilter[1] )

            # Configuramos la función que "animará" nuestra gráfica
            line_ani = animation.FuncAnimation(fig, plotData, fargs=(ax1, ax2, globalDataStream, globalDataFilter),
                interval=50, blit=False)
            
            # Configuramos y lanzamos el hilo encargado de leer datos del serial
            dataCollector = threading.Thread( target = recording, args=(globalDataStream,globalDataFilter,) )
            dataCollector.start()
            plt.show()
            dataCollector.join()

    except KeyboardInterrupt:
        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        audioRecord.terminate()
        print("Exit .... ok!");
        sys.exit()

"""
Function to execute main
"""
if __name__ == "__main__":
    main()