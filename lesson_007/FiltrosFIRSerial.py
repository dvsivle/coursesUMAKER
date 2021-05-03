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
import serial
import time

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
from scipy.fftpack import fftshift

from scipy.signal import butter, lfilter, freqz
from scipy import arange

# Mathematical operations
import numpy as np

# Convert native Python data types
import struct

# Plot

globalStream = []

rows = 2
cols = 1

max_length = 1000
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
# Time record seconds
TIME = 1
# Record at 44100 samples per second
#Fs = 44100
Fs = 44100

# Structure to the stream
stream = audioRecord.open(
    format = FORMAT,
    channels =  CHANNEL,
    rate = Fs,
    input = True,
    output = True,
    frames_per_buffer = FRAMES
    )

def recording(outDatas):
    """
    Recording
    """
    try:
        # Get the stream
        while (True):
            # Get bytes
            data = stream.read(FRAMES) 
            # Get time signal data
            outData = struct.unpack( str(FRAMES)+ "h",data)
            # Get tuple data
            outDatas.append( outData[0] )                        
            if len( outDatas ) > max_length:
                # remove the first element
                outDatas.pop(0)
    except KeyboardInterrupt:
        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        audioRecord.terminate()
        print("Exit .... ok!");
        sys.exit()

def readSerial(outDatas):
    """
    Read Serial
    """
    try:
        arduino = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1.0)
        # Get the stream
        while (True):
            # Serial read section
            msg = arduino.read()
            data = msg.decode()
            values = []

            if( len(data) != 0 ):
                outDatas.append( int(data) )
                        
            if( len( outDatas ) > 50 ):
                outDatas.pop(0)

    except KeyboardInterrupt:
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

def plotFrecuecyDomain(chart,line,data):
    '''
    Plot data in the subplot and set data in the line
    '''
    NFFT = 1024 #NFFT-point DFT      
    
    X = fftshift( fourier.fft(data,NFFT) ) #compute DFT using FFT  
    
    #fVals = np.arange(start = -NFFT/2,stop = NFFT/2)/NFFT #DFT Sample points        
    
    fVals=np.arange(start = -NFFT/2,stop = NFFT/2)*Fs/NFFT

    # Get the limit to axis
    valuesX = fVals 
    valuesY = np.abs( X )

    limitAxisY = max(valuesY) + 10
    limitAxisX = max(valuesX) + 10
    
    # Set the limit to axis
    chart.set_ylim( 0,limitAxisY )
    chart.set_xlim( -1*limitAxisX,limitAxisX )
    
    #chart.set_xlim( -5,5 )
    
    # Prepare data to X
    #valuesX = list( range(0,length) )
    # Set values Y
    line.set_data(valuesX , valuesY)

def plotAmplitudeDomain(chart,line,data):
    '''
    Plot data in the subplot and set data in the line
    '''
    # Get the length data 
    length = len(data)
    # Get the limit to axis
    limitAxisY = max(data) + 10
    limitAxisX = length + 10
    
    # Set the limit to axis
    chart.set_ylim( -1*limitAxisY,limitAxisY )
    chart.set_xlim( 0,limitAxisX )
    # Prepare data to X
    valuesX = list( range(0,length) )
    # Set values Y
    line.set_data(valuesX , data)


def plotData(num,charst,lines,data):
    """
    Plot data
    """
    if( len( data) >= 50 ):
        #print("Data:",data )
        # plot time domain
        plotAmplitudeDomain(charst[0],lines[0],data)

    # Sample rate and desired cutoff frequencies (in Hz).    
    '''
    fs = Fs
    lowcut = 500.0
    highcut = 1250.0

    data_ = data.copy()
    #Butterworth low-pass
    lowPass = butter_lowpass_filter(data_, lowcut,fs, order=6)

    data_ = data.copy()
    #Butterworth high-pass
    highPass = butter_highpass_filter(data_, lowcut,fs, order=6)


    # plot time domain
    plotAmplitudeDomain(charst[0],lines[0],data)
    plotAmplitudeDomain(charst[0],lines[1],lowPass)
    plotAmplitudeDomain(charst[0],lines[2],highPass)

    # plot frecuency    
    plotFrecuecyDomain(charst[1],lines[3],data)
    plotFrecuecyDomain(charst[1],lines[4],lowPass)
    plotFrecuecyDomain(charst[1],lines[5],highPass)
    '''
    return charst,lines


def main():    
    """
    This function implements the estimation of 
    the heart rate by means of digital filters.
    """
    lines = []
    charst = []
    
    fig, (ax1,ax2) = plt.subplots(2)
    charst.append(ax1)
    charst.append(ax2)

    fig.suptitle('Filtros FIR')
    
    charst[0].set_title('Time domain')        
    charst[0].set_ylabel('Amplitude')    
    charst[0].set_xlabel('Time')    
    
    hl1, = ax1.plot( globalStream, globalStream ,color="blue", label='Stream')
    lines.append(hl1)
    
    hl2, = ax1.plot( globalStream, globalStream ,color="green",label='Butterworth low-pass' )
    lines.append(hl2)
    
    hl3, = ax1.plot( globalStream, globalStream,color="red",label='Butterworth high-pass')
    lines.append(hl3)

    hll1, = ax2.plot( globalStream, globalStream ,color="blue", label='Stream')
    lines.append(hll1)
    
    hll2, = ax2.plot( globalStream, globalStream ,color="green",label='Butterworth low-pass')
    lines.append(hll2)
    
    hll3, = ax2.plot( globalStream, globalStream,color="red",label='Butterworth high-pass')
    lines.append(hll3)

    charst[1].set_title('Frequency domain')        
    charst[1].set_ylabel('|Y(f)|')    
    charst[1].set_xlabel('Frequency')    
    
    # Animate matplot
    line_ani = animation.FuncAnimation(fig,
                                       plotData,
                                       fargs=(charst,lines,globalStream),
                                       frames=1,
                                       interval=100,
                                       blit=False
                                       )
    
    # Threading
    dataCollector = threading.Thread( target = readSerial, args=(globalStream,) )
    dataCollector.start()

    plt.legend()
    plt.show()
    dataCollector.join()


if __name__ == "__main__":
    """
    Function to execute main
    """
    main()