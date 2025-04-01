import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import cumulative_trapezoid
import sklearn.model_selection as sk
from tensorflow.keras.models import load_model



# Load data
def load_data():
    """Loads and combines DOS and label data."""
    dos_data_1 = np.load("n12_disorder_level/data_n12_exp.npy")
    label_data_1 = np.load("n12_disorder_level/label_n12_exp.npy")
    dos_data_2 = np.load("n12_disorder_level/data_n12_level.npy")
    label_data_2 = np.load("n12_disorder_level/label_n12_level.npy")
    
    dos_data = np.concatenate((dos_data_2, dos_data_1))
    label_data = np.concatenate((label_data_2, label_data_1))
    return dos_data, label_data

def preprocess_data(dos_data, label_data, n=12, res=250, n_data=1500):
    """Prepares and normalizes data for training."""
    data = np.zeros((n_data, n * res))
    label = np.zeros((n_data, 2 * n - 1))
    
    for i in range(n_data):
        data[i, :] = dos_data[i * n * res:(i + 1) * n * res]
        label[i, :] = label_data[(i * (2 * n - 1) + i * 2):((i + 1) * (2 * n - 1) + i * 2)]
    
    # Subsampling and scaling
    dos_data = data[:, ::2]
    label_data = label + 0.5
    
    scaler_gamma = MinMaxScaler()
    label_data[:, 12:] = scaler_gamma.fit_transform(label[:, 12:])
    
    return dos_data, label_data

def add_noise_and_scale(dos_data, noise_level=0.0001):
    """Adds noise to the data and normalizes it."""
    noise_matrix = np.random.rand(*dos_data.shape) * noise_level
    dos_noisy = dos_data + noise_matrix
    return dos_noisy / np.amax(abs(dos_noisy))

def compute_dIdV(dos_scaled, sample_size, n_site=12, res=125):
    """Computes dI/dV using cumulative integration."""
    omega_values = np.linspace(0.70, 1.32, res)
    dIdV = np.zeros((sample_size, n_site * res))
    
    for i in range(sample_size):
        for l in range(n_site):
            dIdV[i, 1 + l * res:(l + 1) * res] = cumulative_trapezoid(dos_scaled[i, l * res:(l + 1) * res], x=omega_values, dx=0.01)
    
    return dIdV / np.amax(dIdV) * 0.75

def split_data(dIdV_norm, label_data, n=12, res=125):
    """Splits data into training and testing sets, adjusting label size if needed."""
    # Ensure label_data matches dIdV_norm in number of rows
    if label_data.shape[0] * 2 == dIdV_norm.shape[0]:
        label_data = np.repeat(label_data, 2, axis=0)  # Duplicate labels
    
    X_train_all, X_test_all, y_train_all, y_test_all = [], [], [], []
    
    for i in range(n - 2):
        X_features = dIdV_norm[:, i * res:(3 + i) * res]  # Slice features
        y_labels = np.concatenate(
            (label_data[:, i:3 + i] + 0.5, label_data[:, i + n:i + n + 2]),
            axis=1
        )  # Slice labels
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_labels, test_size=0.1
        )

        X_train_all.append(X_train)
        X_test_all.append(X_test)
        y_train_all.append(y_train)
        y_test_all.append(y_test)
    
    return (
        np.concatenate(X_train_all),
        np.concatenate(X_test_all),
        np.concatenate(y_train_all),
        np.concatenate(y_test_all),
    )

def build_ANN(input_dim, output_dim, optimizer='adam'):
    """Builds a simple feedforward neural network."""
    model = keras.Sequential([
        layers.Dense(input_dim, activation='relu'),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1500, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(200, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(5, activation='relu')
    ])
    model.compile(optimizer=optimizer, loss='MeanSquaredError')
    return model

def train_model(model, X_train, y_train, epochs=25, batch_size=32):
    """Trains the neural network model."""
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1)

def plot_loss(history):
    """Plots training and validation loss."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(history.history['loss'], label='Training Loss', color='#007acc', linewidth=2.5, marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#f76c5e', linewidth=2.5, marker='s')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(frameon=True, shadow=True, fontsize=14)
    plt.show()

def plot_predictions(y_test, predictions):
    """Plots predicted vs true values."""
    plt.figure(dpi=150, figsize=(6,4))
    for i in range(3):
        plt.scatter(y_test[:, i], predictions[:, i], label=f'Site {i+1}')
    plt.xlabel("True Value")
    plt.ylabel("Prediction")
    plt.legend()
    plt.title('Intermolecular exchange')
    plt.show()
    
    plt.figure(dpi=150, figsize=(6,4))
    for i in range(2):
        plt.scatter(y_test[:, i+3], predictions[:, i+3], label=f'Site {i+1}')
    plt.xlabel("True Value")
    plt.ylabel("Prediction")
    plt.legend()
    plt.title('Intramolecular exchange')
    plt.show()


def averaged_predictions(data, label, length=14, n_data=20, model_ann=None):
    """
    Compute averaged predictions using an artificial neural network.
    
    Parameters:
        data (ndarray): Input data for predictions.
        label (ndarray): Corresponding labels.
        length (int): Length of prediction window.
        n_data (int): Number of data samples.
        model_ann (keras.Model): Trained ANN model.
    
    Returns:
        ndarray: Averaged prediction values.
    """
    if model_ann is None:
        raise ValueError("A trained model must be provided.")
    
    values_averaged = np.zeros((n_data, length * 2 - 1))
    
    for l in range(0, (length - 4), 1):
        pred_1 = model_ann.predict(data[:, (l+0) * 250:750 + (l+0) * 250])
        pred_2 = model_ann.predict(data[:, (l+1) * 250:750 + (l+1) * 250])
        pred_3 = model_ann.predict(data[:, (l+2) * 250:750 + (l+2) * 250])
        
        values_averaged[:, l+2] = (pred_1[:, 2] + pred_2[:, 1] + pred_3[:, 0]) / 3
        
        if l == 0:
            values_averaged[:, l] = pred_1[:, 0]
            values_averaged[:, l+1] = (pred_1[:, 1] + pred_2[:, 0]) / 2
        else:
            values_averaged[:, l+4] = pred_3[:, 2]
            values_averaged[:, l+3] = (pred_2[:, 2] + pred_3[:, 1]) / 2
    
    for l in range(0, (length - 3), 1):
        pred_1 = model_ann.predict(data[:, (l+0) * 250:750 + (l+0) * 250])
        pred_2 = model_ann.predict(data[:, (l+1) * 250:750 + (l+1) * 250])
        
        values_averaged[:, length + l + 1] = (pred_1[:, 4] + pred_2[:, 3]) / 2
        
        if l == 0:
            values_averaged[:, length + l] = pred_1[:, 3]
        else:
            values_averaged[:, length + l + 2] = pred_2[:, 4]
    
    return values_averaged

# Function to modify dI/dV data for experimental-like behavior
def exp_dIdV(data, offset_value=0.013, gap_value=0.03, noise_value=0.002, res=125):
    """
    Adjusts dI/dV data to simulate experimental-like behavior.
    
    Parameters:
        data (ndarray): Input dI/dV data.
        offset_value (float): Offset applied to data.
        gap_value (float): Superconducting gap value.
        noise_value (float): Noise level added.
        res (int): Resolution.
    
    Returns:
        ndarray: Modified dI/dV data.
    """
    dIdV_test = data + offset_value
    noise_matrix = np.random.rand(res) * noise_value    
    noise_matrix[:20] *= 0.5
    dIdV_noise = dIdV_test + noise_matrix
    return np.nan_to_num(dIdV_noise)

# Prediction Fidelity
def prediction_fidelity(delta_pred, delta_true):
    """
    Compute prediction fidelity between predicted and true values.
    
    Parameters:
        delta_pred (array-like): Predicted values.
        delta_true (array-like): True values.
    
    Returns:
        float: Computed prediction fidelity.
    """
    delta_pred, delta_true = np.array(delta_pred), np.array(delta_true)
    mean_pred, mean_true = np.mean(delta_pred), np.mean(delta_true)
    var_pred, var_true = np.var(delta_pred), np.var(delta_true)
    covariance = np.mean(delta_pred * delta_true) - mean_pred * mean_true
    return np.abs(covariance) / np.sqrt(var_pred * var_true)

# Neural Network Training (Main Code)
def train_dIdV_nn(dIdV_exp, label_data, res=125, n=12):
    """
    Trains a neural network using experimental dI/dV data.
    
    Parameters:
        dIdV_exp (ndarray): Enhanced dI/dV dataset.
        label_data (ndarray): Corresponding labels.
        res (int): Resolution of dI/dV data.
        n (int): Number of training sets.
    
    Returns:
        keras.Model: Trained ANN model.
    """
    label_exp = np.tile(label_data, (2, 1))
    
    X_train_all_exp, y_train_all_exp = [], []
    X_test_all_exp, y_test_all_exp = [], []
    
    for i in range(n-2):     
        X_train, X_test, y_train, y_test = sk.train_test_split(
            dIdV_exp[:, i*res:(i+3)*res],
            np.concatenate((label_exp[:, i:i+3] + 0.5, label_exp[:, i+n:i+n+2]), axis=1),
            test_size=0.1
        )
        X_train_all_exp.append(X_train)
        X_test_all_exp.append(X_test)
        y_train_all_exp.append(y_train)
        y_test_all_exp.append(y_test)
    
    X_train_all = np.concatenate(X_train_all_exp)
    X_test_all = np.concatenate(X_test_all_exp)
    y_train_all = np.concatenate(y_train_all_exp)
    y_test_all = np.concatenate(y_test_all_exp)
    
    model_dIdV_exp = build_ANN()
    model_dIdV_exp.fit(X_train_all, y_train_all, epochs=25, batch_size=64, shuffle=True, validation_split=0.1)
    
    return model_dIdV_exp


def add_linear(data, res=125):
    value = np.mean(data) * np.random.uniform(0.1, 1.2)
    count = 0 
    while data[count] <= value:
        count += 1
    for i in range(int(res - count)):
        data[count + i] = data[count + i] + i * np.random.uniform(1.5, 3) * np.random.uniform(0.000032, 0.000051) 
    return data

def exp_dIdV(data, offset_value=0.013, gap_value=0.03, noise_value=0.002, res=125):
    """
    Adjusts dI/dV data to simulate experimental-like behavior.
    
    Parameters:
        data (ndarray): Input dI/dV data.
        offset_value (float): Offset applied to data.
        gap_value (float): Superconducting gap value.
        noise_value (float): Noise level added.
        res (int): Resolution.
    
    Returns:
        ndarray: Modified dI/dV data.
    """
    dIdV_test = data + offset_value
    noise_matrix = np.random.rand(res) * noise_value    
    noise_matrix[:20] *= 0.5
    dIdV_noise = dIdV_test + noise_matrix
    dIdV_linear = add_linear(dIdV_noise, res)
    return np.nan_to_num(dIdV_linear)

def enhance_dIdV(dIdV, n_enhance=2, res=125, noisy=1):
    """Enhances dI/dV data by adding experimental noise variations."""
    dIdV_exp = np.zeros((1500 * n_enhance, 12 * res))
    label_data_expanded = np.tile(label_data, (n_enhance, 1))
    for a in range(n_enhance):
        offset_value = np.random.uniform(0.009, 0.016)   
        gap_value = np.random.uniform(0.028, 0.032)   
        noise_value = 0.001 * noisy 
        print(f"Enhancement {a+1}: Offset={offset_value:.5f}, Gap={gap_value:.5f}, Noise={noise_value:.5f}")
        
        for b in range(1500):
            for c in range(12):
                dIdV_exp[a * 1500 + b, c * res:(c + 1) * res] = exp_dIdV(abs(dIdV[b, c * res:(c + 1) * res]), noise_value=noise_value)
    
    return dIdV_exp / np.amax(dIdV_exp) * 0.75, label_data_expanded


# Main script execution
if __name__ == "__main__":
    
    # # load and preprocess data
    # dos_data, label_data = load_data()
    # dos_data, label_data = preprocess_data(dos_data, label_data)
    # dos_scaled = add_noise_and_scale(dos_data)
    # dIdV_norm = compute_dIdV(dos_scaled, sample_size=1500)
    # X_train, X_test, y_train, y_test = split_data(dIdV_norm, label_data)
    
    # # (1) build NN based on theory model
    # model = build_ANN(X_train.shape[1], y_train.shape[1])
    # history = train_model(model, X_train, y_train)
    # plot_loss(history)
    
    # # predictions for test data
    # predictions = model.predict(X_test)
    # plot_predictions(y_test, predictions)
    
    # # save model
    # model.save('model_dIdV.keras')


    # (2) build NN based on theory model for realistic dIdV
    dIdV_norm = compute_dIdV(dos_scaled, sample_size=1500)
    dIdV_exp_norm, label_data_expanded = enhance_dIdV(dIdV_norm)
    X_train, X_test, y_train, y_test = split_data(dIdV_exp_norm, label_data_expanded)
    
    model_exp = build_ANN(X_train.shape[1], y_train.shape[1])
    history = train_model(model_exp, X_train, y_train)
    plot_loss(history)
    
    # predictions for test data
    predictions = model_exp.predict(X_test)
    plot_predictions(y_test, predictions)
    
    # save model
    model_exp.save('model_dIdV_exp.keras')

    
