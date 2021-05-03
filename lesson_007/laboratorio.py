"""
@author UMAKER
@about In this code to show plots with
matplotlib
"""

# Packages

import serial 
import time

# system
import sys

#Threading
import threading

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


# Variables

globalStream = []

max_length = 1000
max_axis_Y = 100

def readSerial( outDatas ):
    """
    Read serial inputs/devices
    """
    try:
        arduino = serial.Serial('/dev/ttyUSB0',baudrate=9600,timeout=1.0)
        while (True):
            # Serial read section
            msg = arduino.read()
            data = msg.decode()
            
            if(len(data) != 0):
                outDatas.append( int(data) )
            
            if( len(outDatas) > 50):
                outDatas.pop(0)
    except KeyboardInterrupt:
        print("Exit ... ok!")
        sys.exit()

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
    chart.set_ylim( -1*limitAxisY, limitAxisY)
    chart.set_xlim( 0, limitAxisX)

    # Prepare data to X
    valuesX = list( range(0,length) )

    # Set values Y
    line.set_data( valuesX, data )

def plotData( num, charst,lines,data ):
    """
    Plot data
    """
    if( len(data) >= 50 ):
        plotAmplitudeDomain(charst[0],lines[0],data)
    
    return charst,lines

def main():
    """
    This function implements the plots with matplotlib
    """
    lines  = []  # lines to plot
    charst = []  # chart to show in the window

    fig, (ax1,ax2) = plt.subplots(2)
    
    charst.append(ax1)
    charst.append(ax2)

    fig.subtitle("Serial plot")
    charst[0].set_title('Time domain')
    charst[0].set_ylabel('Amplitude')
    charst[0].set_xlabel(' Time')

    hl1, = ax1.plot( globalStream,globalStream, color="blue",label='Stream' )
    lines.append( hl1 )
    
    # Check this !!!
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

    # Animate matplotlib
    line_ani = animation.FuncAnimation(fig,
    plotData,
    fargs=(charst,lines,globalStream),
    frames=1,
    interval=100,
    blit=False)

    # Threading
    dataCollector = threading.Thread( target = readSerial,args=(globalStream) )
    dataCollector.start()

    plt.legend()
    plt.show()
    dataCollector.join()

if __name__ == "__main__":
    """
    Function to execute main
    """
    main()