# Analysis of Different CLASS_WEIGHT_INCIDENT Values

I compared the data for this case using the LSTM model with:
```
combined_data = np.column_stack((cpu, cpu_diff))
INCIDENT_STEP = 200
CLASS_WEIGHT_INCIDENT = ...
EPOCHS = 15
ALERT_THRESHOLD = 0.5
W, H = 15, 5
N_STEPS = 2000
```

## CLASS_WEIGHT_INCIDENT = 10.0

| Run | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | Normal | 0.97 | 0.99 | 0.98 | 368 |
| | Incident | 0.90 | 0.64 | 0.75 | 28 |
| | Weighted Avg | 0.97 | 0.97 | 0.97 | 396 |
| **#2** | Normal | 0.97 | 1.00 | 0.99 | 368 |
| | Incident | 0.95 | 0.64 | 0.77 | 28 |
| | Weighted Avg | 0.97 | 0.97 | 0.97 | 396 |
| **#3** | Normal | 0.97 | 0.99 | 0.98 | 368 |
| | Incident | 0.90 | 0.64 | 0.75 | 28 |
| | Weighted Avg | 0.97 | 0.97 | 0.97 | 396 |

**Observation:** The model is highly reliable but safe. Incident precision is 90+ and Normal recall is almost perfect. Incident recall is constantly ~0.64. The model fails to explore riskier patterns that might indicate an incident.

---

## CLASS_WEIGHT_INCIDENT = 20.0

| Run | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | Normal | 0.97 | 0.99 | 0.98 | 368 |
| | Incident | 0.90 | 0.64 | 0.75 | 28 |
| | Weighted Avg | 0.97 | 0.97 | 0.97 | 396 |
| **#2** | Normal | 0.97 | 1.00 | 0.99 | 368 |
| | Incident | 0.95 | 0.64 | 0.77 | 28 |
| | Weighted Avg | 0.97 | 0.97 | 0.97 | 396 |
| **#3** | Normal | 0.98 | 0.99 | 0.99 | 368 |
| | Incident | 0.91 | 0.71 | 0.80 | 28 |
| | Weighted Avg | 0.97 | 0.97 | 0.97 | 396 |

**Observation:** This system is also stable: Incident precision remains 90+ and Normal recall is nearly perfect. Incident recall is usually ~0.64, but it occasionally performs better. This weight provides the best balance between Precision and Recall.

---

## CLASS_WEIGHT_INCIDENT = 30.0

| Run | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | Normal | 0.97 | 0.94 | 0.96 | 368 |
| | Incident | 0.45 | 0.64 | 0.53 | 28 |
| | Weighted Avg | 0.94 | 0.92 | 0.93 | 396 |
| **#2** | Normal | 0.97 | 0.92 | 0.94 | 368 |
| | Incident | 0.38 | 0.64 | 0.47 | 28 |
| | Weighted Avg | 0.93 | 0.90 | 0.91 | 396 |
| **#3** | Normal | 0.97 | 0.92 | 0.95 | 368 |
| | Incident | 0.39 | 0.64 | 0.49 | 28 |
| | Weighted Avg | 0.93 | 0.90 | 0.91 | 396 |
| **#4** | Normal | 0.96 | 0.62 | 0.75 | 368 |
| | Incident | 0.11 | 0.64 | 0.19 | 28 |
| | Weighted Avg | 0.90 | 0.62 | 0.71 | 396 |
| **#5** | Normal | 0.98 | 0.97 | 0.98 | 368 |
| | Incident | 0.66 | 0.75 | 0.70 | 28 |
| | Weighted Avg | 0.96 | 0.95 | 0.96 | 396 |
| **#6** | Normal | 0.98 | 0.90 | 0.94 | 368 |
| | Incident | 0.36 | 0.71 | 0.48 | 28 |
| | Weighted Avg | 0.93 | 0.89 | 0.91 | 396 |

**Observation:** This system becomes unstable: Incident precision drops significantly and Normal recall varies. Incident recall is often better but it triggers too many false alarms because the model becomes over-sensitive and aggressive.

---

### Conclusion
**CLASS_WEIGHT_INCIDENT = 20.0** is the best choice. It triggers fewer false alarms than weight 30.0, but is more capable of catching incidents than weight 10.0.
