
"""
@author Noe VG
@about Working with matplotlib
"""

# system
#import sys

# Plot
import matplotlib.pyplot as plt

# Mathematical operations & structures
import numpy as np

"""
------------
| x ---  y |
|----------|
| 1  |  10 |
| 2  |   8 |
| 3  |   9 |
| 4  |   7 |
| 5  |   7 |
| 6  |   1 |
| 7  |   6 |
| 8  |   6 |
| 9  |   4 |
| 10 |   3 |
------------
"""

def codeA():
    #try catch
    try:
        # Data to axis Y
        data = [10,8,9,7,7,1,6,6,4,3]

        # Create figure
        fig = plt.figure()

        # Generate data to axis X
        dataX = np.arange( len(data) )

        # Plot data in the charts
        plt.plot(dataX,data)

        #Show the plot
        plt.show()
    except KeyboardInterrupt:
        print("Exit ... ok!!!")
        #sys.exit()

def codeB():
    #try catch
    try:
        # Data to axis Y
        dataA = [10,8,9,7,7,1,6,6,4,3]
        dataB = [10,4,2,9,7,1,6,6,6,3]

        # Create figure
        fig = plt.figure()

        # Generate data to axis X
        dataXa = np.arange( len(dataA) )

        dataXb = np.arange( len(dataB) )

        # Plot data in the charts
        plt.plot(dataXa,dataA,label="blue")
        plt.plot(dataXb,dataB,label="naranja")

        #Show the plot
        plt.legend()
        plt.show()
    except KeyboardInterrupt:
        print("Exit ... ok!!!")
        #sys.exit()

def codeC():
    #try catch
    try:
        # Data to axis Y
        dataA = [10,8,9,7,7,1,6,6,4,3]
        dataB = [10,4,2,9,7,1,6,6,6,3]

        # Create figure
        fig, (ax1,ax2) = plt.subplots(2)

        # Generate data to axis X
        dataXa = np.arange( len(dataA) )

        dataXb = np.arange( len(dataB) )

        # Plot data in the charts
        ax1.set_title("Data A")
        ax1.plot(dataXa,dataA,label = "list A", color="pink")
        ax1.legend()
        ax1.set_xlabel("Values X")
        ax1.set_ylabel("Values Y")

        ax2.set_title("Data B")
        ax2.plot(dataXb,dataB,label = "list B", color ="green")
        ax2.legend()
        ax2.set_xlabel("Values X")
        ax2.set_ylabel("Values Y")


        #Show the plot
        plt.legend()
        plt.show()

    except KeyboardInterrupt:
        print("Exit ... ok!!!")
        #sys.exit()

def main():
    codeC()

if __name__ == "__main__":
    """
    Function to execute the main
    """
    main()
