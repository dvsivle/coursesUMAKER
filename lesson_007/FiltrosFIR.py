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

globalStream = []

rows = 2
cols = 1

max_axis_X = 100
max_axis_Y = 100

# Create an interface to PortAudio
audioRecord = pyaudio.PyAudio()

# Values to recording

# Record in chunks of 1024 times six samples
#FRAMES = 1024*6
FRAMES = 1

# Format bites per sample
FORMAT = pyaudio.paInt16
# Chanell to get
CHANNEL = 1
# Record at 44100 samples per second
SAMPLES = 44100
"""
Recording
"""
def recording(outDatas):    
    # Get the stream
    stream = audioRecord.open(
        format = FORMAT,
        channels =  CHANNEL,
        rate = SAMPLES,
        input = True,
        output = True,
        frames_per_buffer = FRAMES
        )
    try:
        while (True):
            # Get bytes
            data = stream.read(FRAMES) 
            # Get time signal data
            outData = struct.unpack( str(FRAMES)+ "h",data)
            outDatas.append(outData[0])
            #i = input()
            if len( outDatas ) > 100:
                outDatas.pop(0)
    
    except KeyboardInterrupt:
        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        audioRecord.terminate()
        print("Exit .... ok!");
        sys.exit()

"""
Plot data
"""
def plotData(num,ax1,hl1,data):
    valuesX = list( range(0,len(data)) )
    maxAxisX = max_axis_X + 200
    maxAxisY = max(data) + 200
    
    ax1.set_ylim( -1*maxAxisY,maxAxisY)
    hl1.set_data(valuesX , data)
    return hl1

"""
This function implements the estimation of 
the heart rate by means of digital filters.
"""
def main():    
    fig, (ax1,ax2) = plt.subplots(2)
    fig.suptitle('Filtros FIR')
    
    ax1.set_title('Real Stream')    
    ax1.set_ylim(-1*max_axis_Y,max_axis_Y)   
    ax1.set_xlim(1,max_axis_X)
    hl1, = ax1.plot( globalStream, globalStream )

    ax2.set_title('Filter')    
    ax2.set_ylim(-1*max_axis_Y,max_axis_Y)   
    ax2.set_xlim(1,max_axis_X)
    hl2, = ax1.plot( globalStream, globalStream )


    # Animate matplot
    line_ani = animation.FuncAnimation(fig,
                                       plotData,
                                       fargs=(ax1,hl1, globalStream),
                                       frames=1,
                                       interval=100,
                                       blit=False
                                       )
    
    # Threading
    dataCollector = threading.Thread( target = recording, args=(globalStream,) )
    dataCollector.start()
    plt.show()
    dataCollector.join()

"""
Function to execute main
"""
if __name__ == "__main__":
    main()