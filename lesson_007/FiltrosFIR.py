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
from scipy import fft, arange
from scipy import signal
import scipy
from scipy.signal import butter, lfilter
from scipy.signal import freqz

# Mathematical operations
import numpy as np

# Convert native Python data types
import struct

# Plot

globalStream = []


rows = 2
cols = 1

max_axis_X = 1000
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
#Fs = 44100
Fs = 44100

"""
Recording
"""
def recording(outDatas):    
    # Get the stream
    # Strcture to the stream
    stream = audioRecord.open(
        format = FORMAT,
        channels =  CHANNEL,
        rate = Fs,
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
            outDatas.append( outData[0] )
                        
            #i = input()
            if len( outDatas ) > max_axis_X:
                outDatas.pop(0)
                
    
    except KeyboardInterrupt:
        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        audioRecord.terminate()
        print("Exit .... ok!");
        sys.exit()


def butter_lowpass(cutoff, fs, order=5):
    """
    Design lowpass filter.

    Args:
        - cutoff (float) : the cutoff frequency of the filter.
        - fs     (float) : the sampling rate.
        - order    (int) : order of the filter, by default defined to 5.
    """
    # calculate the Nyquist frequency
    nyq = 0.5 * fs

    # design filter
    low = cutoff / nyq
    b, a = butter(order, low, btype='low', analog=False)

    # returns the filter coefficients: numerator and denominator
    return b, a 

def butter_highpass(cutoff, fs, order=5):
    """
    Design a highpass filter.

    Args:
        - cutoff (float) : the cutoff frequency of the filter.
        - fs     (float) : the sampling rate.
        - order    (int) : order of the filter, by default defined to 5.
    """
    # calculate the Nyquist frequency
    nyq = 0.5 * fs

    # design filter
    high = cutoff / nyq
    b, a = butter(order, high, btype='high', analog=False)

    # returns the filter coefficients: numerator and denominator
    return b, a 

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data,cutoff,fs, order=5):
    b, a = butter_lowpass(cutoff,fs,order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data,cutoff,fs, order=5):
    b, a = butter_highpass(cutoff,fs,order=order)
    y = lfilter(b, a, data)
    return y
def getFrecuecyDomain(data):
    x = data - np.average(data)  # zero-centering

    n = len(x)
    k = arange(n)
    tarr = n / float(Fs)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fourier.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]
   
    return  frqarr, x
"""
Plot data
"""
def plotData(num,ax1,ax2,hl1,hl2,hl3,hll1,hll2,hll3,data):
    
    # plot time domain
    valuesX = list( range(0,len(data)) )
    maxAxisY = max(data) + 200
    
    ax1.set_ylim( -1*maxAxisY,maxAxisY)
    hl1.set_data(valuesX , data)
    hl1.set_color("blue")
    
    # plot frecuency
    
    data_ = data.copy()
    dataX, dataY = getFrecuecyDomain(data_)
    maxAxisY = max(dataY) + 1
    maxAxisX = max(dataX)
    ax2.set_xlim(0, maxAxisX )
    ax2.set_ylim( 0, maxAxisY )
    
    hll1.set_data( dataX,abs(dataY))
    hll1.set_color("blue")

    # Sample rate and desired cutoff frequencies (in Hz).    
    fs = Fs
    lowcut = 500.0
    highcut = 1250.0

    #datay = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)
    #print(datay)

    data_ = data.copy()
    #Butterworth low-pass
    lowPass = butter_lowpass_filter(data_, lowcut,fs, order=6)

    valuesX = list( range(0,len(lowPass)) )
    maxAxisY = max(lowPass) + 200
    hl2.set_data(valuesX, lowPass)
    hl2.set_color("orange")

    data_ = lowPass.copy()
    dataX, dataY = getFrecuecyDomain(data_)
    maxAxisY = max(dataY) + 1
    maxAxisX = max(dataX)
    ax2.set_xlim(0, maxAxisX )
    ax2.set_ylim( 0, maxAxisY )
    
    hll2.set_data( dataX,abs(dataY))
    hll2.set_color("orange")


    data_ = data.copy()
    #Butterworth high-pass
    highPass = butter_highpass_filter(data_, lowcut,fs, order=6)

    valuesX = list( range(0,len(highPass)) )
    maxAxisY = max(highPass) + 200
    hl3.set_data(valuesX, highPass)
    hl3.set_color("green")

    data_ = highPass.copy()
    dataX, dataY = getFrecuecyDomain(data_)
    maxAxisY = max(dataY) + 1
    maxAxisX = max(dataX)
    ax2.set_xlim(0, maxAxisX )
    ax2.set_ylim( 0, maxAxisY )
    
    hll3.set_data( dataX,abs(dataY))
    hll3.set_color("green")

    return hl1,hl2,hl3,hll1,hll2,hll3

"""
This function implements the estimation of 
the heart rate by means of digital filters.
"""
def main():    
    fig, (ax1,ax2) = plt.subplots(2)
    fig.suptitle('Filtros FIR')
    
    ax1.set_title('Time domain')        
    ax1.set_ylabel('Amplitude')    
    ax1.set_xlabel('Time')    
    
    ax1.set_ylim(-1*max_axis_Y,max_axis_Y)   
    ax1.set_xlim(1,max_axis_X)

    hl1, = ax1.plot( globalStream, globalStream ,color="blue", label='Stream')
    hl2, = ax1.plot( globalStream, globalStream)
    hl3, = ax1.plot( globalStream, globalStream)

    ax2.set_title('Frequency domain')        
    ax2.set_ylabel('|Y(f)|')    
    ax2.set_xlabel('Frequency')    
    
    ax2.set_ylim(-1*max_axis_Y,max_axis_Y)   
    ax2.set_xlim(1,max_axis_X)
    hll1, = ax2.plot( globalStream, globalStream,color="blue",label='Stream')
    hll2, = ax2.plot( globalStream, globalStream,color="orange",label='Butterworth low-pass')
    hll3, = ax2.plot( globalStream, globalStream,color="green",label='Butterworth high-pass')

    # Animate matplot
    line_ani = animation.FuncAnimation(fig,
                                       plotData,
                                       fargs=(ax1,ax2,hl1,hl2,hl3,hll1,hll2,hll3,globalStream),
                                       frames=1,
                                       interval=100,
                                       blit=False
                                       )
    
    # Threading
    dataCollector = threading.Thread( target = recording, args=(globalStream,) )
    dataCollector.start()

    plt.legend()
    plt.show()
    dataCollector.join()

"""
Function to execute main
"""
if __name__ == "__main__":
    main()