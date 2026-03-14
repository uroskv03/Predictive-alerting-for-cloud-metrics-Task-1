## Analysis of Different CLASS_WEIGHT_INCIDENT Values

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

### CLASS_WEIGHT_INCIDENT = 10.0

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

### CLASS_WEIGHT_INCIDENT = 20.0

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

### CLASS_WEIGHT_INCIDENT = 30.0

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


## Impact of Prediction Horizon (H) on Incident Recall

The **Incident Recall** is directly affected by the prediction horizon (**H**). Our tests showed a clear inverse relationship between the length of the horizon and the model's performance:

*   **Lower Horizon (H = 2 or 3):** Increases Recall (e.g. **H = 3** -> **Recall ~0.75** ;**H = 2** -> **Recall ~0.82**). This is expected because predicting the immediate future is significantly easier for the model.
*   **Higher Horizon (H = 10 or 20):** Decreases Recall (e.g., **H = 10** -> **~0.47**; **H = 20** -> **Recall ~0.31**). It is much harder for the model to see an incident that is far away in the future if there are no early warning signs.

**Note on Recall Cap:**
In our current setup, the Recall is often capped at a specific value (**0.64** for our standard parameters). This is due to the nature of our data generator: the incident is modeled as a sudden jump without a "slow rise" or pre-warning signal. The model often only realizes an incident is happening at the exact moment it starts.

---

## Comparison: CPU-only vs. CPU + CPU_diff

I compared the performance of a model using only raw **CPU** metrics against one using both **CPU** and **CPU_diff**. While they behave similarly due to the simplicity of the data, but there are several key differences:

The "CPU-only" model is slightly more stable because it has fewer parameters. However, it requires a higher `CLASS_WEIGHT_INCIDENT` to occasionally boost the Incident Recall.

In extensive testing, the "CPU-only" model occasionally dropped below the 0.64 Recall threshold (falling to **0.61** or **0.63**). In contrast, the model with both metrics (**CPU + CPU_diff**) proved to be more consistent and never dropped below the **0.64** threshold.

**Conclusion:**
In general, when comparing the results, both approaches are very similar. However, I give a slight advantage to the model with **CPU_diff** because it explicitly highlights the value change.


## Random Forest vs. LSTM:

Using the **Random Forest** algorithm proved to be more efficient than **LSTM** for this specific task. Several advantages were:

Parameter tuning was more straightforward. Even with `CLASS_WEIGHT_INCIDENT = 1`, the model achieved a baseline Incident recall of **0.64**.

Since it is not a deep-learning model, Random Forest trained faster with better stability.

Adding the cpu_diff showed no visible change, but the reason could be the simplicity of the example.


## Correlation: RAM and CPU Spikes

We introduced **RAM** and **ram_diff** metrics to simulate a relationship where a memory error might trigger a CPU spike. 

Every second CPU incident was triggered by a RAM spike that happened slightly before.

The Incident Recall increased to **0.82**, as expected theoretically: $1 \times 0.5 + 0.64 \times 0.5 = 0.82$



