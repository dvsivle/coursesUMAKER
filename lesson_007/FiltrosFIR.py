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
# sys
import sys

# Capture audio
import pyaudio as pyaudio

# Plot charts
import matplotlib.pyplot as plt

# Library used for scientific computing
import scipy.fftpack as fourier
from scipy import signal

# Mathematical operations
import numpy as np

# Convert native Python data types
import struct

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
def recording():
    print("Recording...")
    
    return audioRecord.open(
        format = FORMAT,
        channels =  CHANNEL,
        rate = SAMPLES,
        input = True,
        output = True,
        frames_per_buffer = FRAMES
        )
    

"""
This function implements the estimation of 
the heart rate by means of digital filters.
"""
def main():    
    # Cancel execute with a key input with ctrl + c
    try:
        while True:
            # Get the stream
            stream = recording()
            # Get bytes
            data = stream.read(FRAMES) 
            # Get time signal data
            dataInt = struct.unpack( str(FRAMES)+ "h",data)
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