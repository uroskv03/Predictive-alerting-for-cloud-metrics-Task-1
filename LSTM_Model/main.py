import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

INCIDENT_STEP = 200
CLASS_WEIGHT_INCIDENT = 5.0
EPOCHS = 15
ALERT_THRESHOLD = 0.5
W, H = 15, 5
N_STEPS = 2000


def generate_data(n_steps, step_freq):
    time = np.linspace(0, n_steps / 20, n_steps)  #100

    cpu_data = np.sin(time) + np.random.normal(0, 0.1, n_steps)
    ram_data = (time * 0.05) + np.random.normal(0, 0.1, n_steps)

    for i in range(100, n_steps, step_freq):
        cpu_data[i:i + 10] += 5.0
    for i in range(90, n_steps, 2 * step_freq):
        ram_data[i:i + 10] += 3.0
    for i in range(80, n_steps, int(step_freq / 10)):
        ram_data[i:i + 10] += 1.0

    return cpu_data, ram_data


def create_windows(data, W, H, threshold=3.0):
    X, y = [], []
    for i in range(len(data) - W - H):
        window = data[i: i + W]
        X.append(window)
        future_window = data[i + W: i + W + H, 0]
        incident_happened = 1 if np.any(future_window > threshold) else 0
        y.append(incident_happened)
    return np.array(X), np.array(y)


if __name__ == '__main__':


    cpu, ram = generate_data(N_STEPS, INCIDENT_STEP)
    cpu_diff = np.diff(cpu, prepend=cpu[0])
    ram_diff = np.diff(ram, prepend=ram[0])
    scaler = StandardScaler()

    #combined_data = scaler.fit_transform(cpu.reshape(-1, 1))
    combined_data = np.column_stack((cpu, cpu_diff))
    #combined_data = np.column_stack((cpu, cpu_diff, ram_diff))
    #combined_data = np.column_stack((cpu, cpu_diff, ram, ram_diff))
    scaled_data = scaler.fit_transform(combined_data)

    X, y = create_windows(scaled_data, W, H)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    model = tf.keras.Sequential([
        #tf.keras.layers.LSTM(32, input_shape=(W, 1)),
        tf.keras.layers.LSTM(32, input_shape=(W, 2)),
        #tf.keras.layers.LSTM(32, input_shape=(W, 3)),
        #tf.keras.layers.LSTM(32, input_shape=(W, 4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'),])

    class_weights = {0: 1.0, 1: CLASS_WEIGHT_INCIDENT}
    print(f"Experiment started: Step={INCIDENT_STEP}, Weight={CLASS_WEIGHT_INCIDENT}, Epochs={EPOCHS}")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, class_weight=class_weights, verbose=1)


    predictions = model.predict(X_test)

    y_pred_bool = (predictions > ALERT_THRESHOLD).astype(int)

    print("\n--- EVALUATION REPORT ---")
    print(classification_report(y_test, y_pred_bool, target_names=['Normal', 'Incident']))

    # 6. Visualization
    plt.figure(figsize=(14, 7))
    view_limit = 400

    plt.subplot(2, 1, 1)
    # Raw data
    offset = len(cpu) - len(y_test)
    plt.plot(cpu[offset:][-view_limit:], label='CPU Utilization (AWS CloudWatch)', color='teal', alpha=0.7)
    plt.plot(ram[offset:][-view_limit:], label='RAM Utilization (AWS CloudWatch)', color='orange', alpha=0.7)
    plt.title(f"Cloud Metrics (Test Period) - Step Frequency: {INCIDENT_STEP}")
    plt.legend()

    # Prediction(red) vs Reality(blue)
    plt.subplot(2, 1, 2)
    plt.plot(y_test[-view_limit:], label='Actual Incident', color='blue', linewidth=2)
    plt.plot(predictions[-view_limit:], label='Model Risk Probability', color='red', linestyle='--', alpha=0.8)
    plt.axhline(y=ALERT_THRESHOLD, color='green', linestyle=':', label='Alert Trigger Threshold')
    plt.title(f"Predictive Alerting Analysis (Recall vs False Positives)")
    plt.ylim(-0.1, 1.1)
    plt.legend()

    plt.tight_layout()
    plt.show()
