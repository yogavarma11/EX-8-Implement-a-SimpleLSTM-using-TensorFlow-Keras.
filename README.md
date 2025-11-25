# EX-8-Implement-a-SimpleLSTM-using-TensorFlow-Keras.
## AIM:
To Implement-a-SimpleLSTM-using-TensorFlow-Keras

## ALGORITHM:
Step 1 — Collect and Prepare Sequential Data

Sequential or time-series data is gathered and preprocessed. The data is scaled (e.g., between 0 and 1) because LSTMs work better with normalized values.

Step 2 — Convert the Data into Input Sequences

The dataset is divided into small overlapping sequences. For example, the last N values are used to predict the next value. The data is reshaped into (samples, timesteps, features) to match LSTM input format.

Step 3 — Build the LSTM Model Using TensorFlow–Keras

The model is compiled using an optimizer like Adam and a loss function like MSE.

Step 4 — Train the LSTM Model

The model is trained using the training sequences. During training, the LSTM learns long-term dependencies using its memory cell and gates. Validation data is used to monitor performance.

Step 5 — Evaluate and Perform Predictions

The model is tested on unseen sequences. Prediction results are compared with actual values to measure accuracy. The trained LSTM can now forecast future time steps or classify new sequences.

Step 6 — Visualize and Analyze Results

Graphs such as predicted vs. actual values and training loss curves are plotted. This helps understand how well the model learned the pattern and whether adjustments are needed.


## PROGRAM:

Name:YOGAVARMA B

Register Number:2305002029

~~~
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def generate_sine_wave(n_points=2000, freq=0.01, noise=0.0):
    x = np.arange(n_points)
    y = np.sin(2 * np.pi * freq * x) + (np.random.normal(scale=noise, size=n_points) if noise > 0 else 0)
    return y

series = generate_sine_wave(n_points=2000, freq=0.005, noise=0.02)

def create_sequences(data, window_size):
    """
    Convert the 1D data array into (X, y) sequences for supervised learning.
    X shape: (samples, window_size, 1)
    y shape: (samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    X = np.array(X)
    y = np.array(y)
    # add feature dimension
    X = X[..., np.newaxis]
    return X, y

WINDOW_SIZE = 50
X, y = create_sequences(series, WINDOW_SIZE)

# split into train / val / test
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, shuffle=False)

print("Shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("  X_test: ", X_test.shape, "y_test: ", y_test.shape)

def build_model(window_size, n_units=64):
    model = Sequential([
        LSTM(n_units, input_shape=(window_size, 1)),
        Dense(1)  # predict next scalar value
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(WINDOW_SIZE, n_units=64)
model.summary()

# -----------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=2
)

eval_result = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss (MSE): {eval_result[0]:.6f}, Test MAE: {eval_result[1]:.6f}")

# Predict next values for the test set (one-step prediction)
y_pred = model.predict(X_test)

def plot_predictions(y_true, y_pred, n_points=200):
    """
    Plot a slice of true vs predicted values.
    """
    plt.figure(figsize=(12,5))
    idx = np.arange(len(y_true))
    plt.plot(idx[:n_points], y_true[:n_points], label='True', linewidth=1.5)
    plt.plot(idx[:n_points], y_pred[:n_points], label='Predicted', linewidth=1.2)
    plt.title("True vs Predicted (first {} points of test set)".format(n_points))
    plt.xlabel("Time step (test index)")
    plt.ylabel("Series value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_predictions(y_test, y_pred, n_points=300)

def rolling_forecast(model, seed_sequence, n_steps):
    """
    Given a seed_sequence of length WINDOW_SIZE, predict n_steps into the future
    by repeatedly appending predictions and using the last WINDOW_SIZE values as input.
    """
    seq = seed_sequence.copy().tolist()
    preds = []
    for _ in range(n_steps):
        x_in = np.array(seq[-WINDOW_SIZE:])[np.newaxis, ..., np.newaxis]  # shape (1, WINDOW_SIZE, 1)
        next_val = model.predict(x_in)[0,0]
        preds.append(next_val)
        seq.append(next_val)
    return np.array(preds)

seed = X_test[0].squeeze()  # first test window as seed
future_preds = rolling_forecast(model, seed, n_steps=200)

# plot seed + future predictions
plt.figure(figsize=(12,4))
plt.plot(np.arange(WINDOW_SIZE), seed, label='Seed (last observed window)')
plt.plot(np.arange(WINDOW_SIZE, WINDOW_SIZE + len(future_preds)), future_preds, label='Rolling forecast')
plt.axvline(WINDOW_SIZE-0.5, color='k', linestyle='--', linewidth=0.8)
plt.title("Rolling multi-step forecast")
plt.xlabel("Time step (relative)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
~~~

## OUTPUT:
### EPOCH
<img width="1170" height="438" alt="image" src="https://github.com/user-attachments/assets/5c297f13-b6f3-481a-bb78-0291e4f78ca5" />

### TEST LOSS(MSE):
<img width="793" height="133" alt="image" src="https://github.com/user-attachments/assets/3992e119-2e52-4295-b73d-fc9cd3c7a2f8" />

## PLOT PREDICTIONS:
<img width="1099" height="512" alt="image" src="https://github.com/user-attachments/assets/3cd6d6d4-a870-4c1b-80a4-31529928dc91" />

### ROLLING FORECAST:
<img width="1026" height="341" alt="image" src="https://github.com/user-attachments/assets/3f7251d4-2598-4a01-b2ab-489439b74ad4" />

## RESULT:

Thus, the Implement-a-SimpleLSTM-using-TensorFlow-Keras is successfully executed.
# EX-8-Implement-a-SimpleLSTM-using-TensorFlow-Keras.
## AIM:
To Implement-a-SimpleLSTM-using-TensorFlow-Keras

## ALGORITHM:
Step 1 — Collect and Prepare Sequential Data

Sequential or time-series data is gathered and preprocessed. The data is scaled (e.g., between 0 and 1) because LSTMs work better with normalized values.

Step 2 — Convert the Data into Input Sequences

The dataset is divided into small overlapping sequences. For example, the last N values are used to predict the next value. The data is reshaped into (samples, timesteps, features) to match LSTM input format.

Step 3 — Build the LSTM Model Using TensorFlow–Keras

The model is compiled using an optimizer like Adam and a loss function like MSE.

Step 4 — Train the LSTM Model

The model is trained using the training sequences. During training, the LSTM learns long-term dependencies using its memory cell and gates. Validation data is used to monitor performance.

Step 5 — Evaluate and Perform Predictions

The model is tested on unseen sequences. Prediction results are compared with actual values to measure accuracy. The trained LSTM can now forecast future time steps or classify new sequences.

Step 6 — Visualize and Analyze Results

Graphs such as predicted vs. actual values and training loss curves are plotted. This helps understand how well the model learned the pattern and whether adjustments are needed.


## PROGRAM:

Name:KAVIPRIYA SP

Register Number:2305002011

~~~
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def generate_sine_wave(n_points=2000, freq=0.01, noise=0.0):
    x = np.arange(n_points)
    y = np.sin(2 * np.pi * freq * x) + (np.random.normal(scale=noise, size=n_points) if noise > 0 else 0)
    return y

series = generate_sine_wave(n_points=2000, freq=0.005, noise=0.02)

def create_sequences(data, window_size):
    """
    Convert the 1D data array into (X, y) sequences for supervised learning.
    X shape: (samples, window_size, 1)
    y shape: (samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    X = np.array(X)
    y = np.array(y)
    # add feature dimension
    X = X[..., np.newaxis]
    return X, y

WINDOW_SIZE = 50
X, y = create_sequences(series, WINDOW_SIZE)

# split into train / val / test
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, shuffle=False)

print("Shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("  X_test: ", X_test.shape, "y_test: ", y_test.shape)

def build_model(window_size, n_units=64):
    model = Sequential([
        LSTM(n_units, input_shape=(window_size, 1)),
        Dense(1)  # predict next scalar value
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(WINDOW_SIZE, n_units=64)
model.summary()

# -----------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=2
)

eval_result = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss (MSE): {eval_result[0]:.6f}, Test MAE: {eval_result[1]:.6f}")

# Predict next values for the test set (one-step prediction)
y_pred = model.predict(X_test)

def plot_predictions(y_true, y_pred, n_points=200):
    """
    Plot a slice of true vs predicted values.
    """
    plt.figure(figsize=(12,5))
    idx = np.arange(len(y_true))
    plt.plot(idx[:n_points], y_true[:n_points], label='True', linewidth=1.5)
    plt.plot(idx[:n_points], y_pred[:n_points], label='Predicted', linewidth=1.2)
    plt.title("True vs Predicted (first {} points of test set)".format(n_points))
    plt.xlabel("Time step (test index)")
    plt.ylabel("Series value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_predictions(y_test, y_pred, n_points=300)

def rolling_forecast(model, seed_sequence, n_steps):
    """
    Given a seed_sequence of length WINDOW_SIZE, predict n_steps into the future
    by repeatedly appending predictions and using the last WINDOW_SIZE values as input.
    """
    seq = seed_sequence.copy().tolist()
    preds = []
    for _ in range(n_steps):
        x_in = np.array(seq[-WINDOW_SIZE:])[np.newaxis, ..., np.newaxis]  # shape (1, WINDOW_SIZE, 1)
        next_val = model.predict(x_in)[0,0]
        preds.append(next_val)
        seq.append(next_val)
    return np.array(preds)

seed = X_test[0].squeeze()  # first test window as seed
future_preds = rolling_forecast(model, seed, n_steps=200)

# plot seed + future predictions
plt.figure(figsize=(12,4))
plt.plot(np.arange(WINDOW_SIZE), seed, label='Seed (last observed window)')
plt.plot(np.arange(WINDOW_SIZE, WINDOW_SIZE + len(future_preds)), future_preds, label='Rolling forecast')
plt.axvline(WINDOW_SIZE-0.5, color='k', linestyle='--', linewidth=0.8)
plt.title("Rolling multi-step forecast")
plt.xlabel("Time step (relative)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
~~~

## OUTPUT:
### EPOCH
<img width="1170" height="438" alt="image" src="https://github.com/user-attachments/assets/5c297f13-b6f3-481a-bb78-0291e4f78ca5" />

### TEST LOSS(MSE):
<img width="793" height="133" alt="image" src="https://github.com/user-attachments/assets/3992e119-2e52-4295-b73d-fc9cd3c7a2f8" />

## PLOT PREDICTIONS:
<img width="1099" height="512" alt="image" src="https://github.com/user-attachments/assets/3cd6d6d4-a870-4c1b-80a4-31529928dc91" />

### ROLLING FORECAST:
<img width="1026" height="341" alt="image" src="https://github.com/user-attachments/assets/3f7251d4-2598-4a01-b2ab-489439b74ad4" />

## RESULT:

Thus, the Implement-a-SimpleLSTM-using-TensorFlow-Keras is successfully executed.
