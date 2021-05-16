"""
@uthor TensorFlow&Keras | Deeep Learning 
"""

#Import packages

# Read data CSV and read structure
import pandas as pd

# Plot charts
import seaborn as sns
import matplotlib.pyplot as plt

# TensorFlow with Keras
import tensorflow as tf

# System
import sys

def working():
    '''
    This function execute the all neuronal with TensorFlow
    '''
    # Read and Shown data
    temperature_df = pd.read_csv("data/data.csv")

    # Shown
    #sns.scatterplot(temperature_df['Celsius'],temperature_df['Fahrenheit'])
    #plt.show()

    # Load Set

    X_train = temperature_df['Celsius'] 
    y_train = temperature_df['Fahrenheit']

    # Create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1,input_shape=[1]))

    # Model Compiler
    model.compile(optimizer=tf.keras.optimizers.Adam(1),loss='mean_squared_error') 

    # Train
    epochs_hist = model.fit(X_train,y_train,epochs=100)
    
    print("Evaluacion para le modelo en su entrenamiento")
    print("Keys:")
    print(epochs_hist.history.keys())

    plt.plot(epochs_hist.history['loss'])
    plt.title('Perdida en su Entrenamiento para nuestro Modelo')
    plt.xlabel('Epoch')
    plt.ylabel('Peridada de entrenamieto')
    plt.legend('Perdida de Entrenamiento')
    plt.show()

    # Prediccion
    Temp_C = 0
    Temp_F = model.predict( [Temp_C] )

    print("Prediccion de Temperatura: "+ str(Temp_F))

    Temp_F = 9/5 * Temp_C + 32
    print("Temperata correcta por Ecuación: ",Temp_F)

    pass

def main():
    '''
    This function init the all working
    '''
    working()

if __name__ == "__main__":
    '''
    This function execute main function
    '''
    main()
