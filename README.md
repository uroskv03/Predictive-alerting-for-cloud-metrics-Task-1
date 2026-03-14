# Predictive Alerting for Cloud Metrics

## Overview

The goal of this project is to predict service incidents within a future horizon (H) based on a look-back window (W) of time-series metrics. 
To evaluate the model, we generate synthetic data:
**CPU Utilization:** Modeled as a sine wave with added noise.
**RAM Utilization:** Modeled with a linear upward trend plus noise and spikes.
**Incidents:** Defined as sudden, simultaneous spikes in both metrics. In our setup, an incident is flagged if the CPU crosses a threshold within the next H steps.

## Additional metrics

In addition to raw metrics, we introduced cpu_diff. This metric serves to further emphasize the velocity of CPU change because it is the most important parameter for detecting the beginning of the jump.

A memory spike can cause a CPU spike, so the ram and ram_diff metrics have been added. Using ram_diff helps stabilize the input by focusing only on the sudden changes.

## Modeling Approaches

 **LSTM (Long Short-Term Memory):** Uses previous values ​​to predict the next ones.

 **Random Forest:** Uses multiple decision trees to classify potential incidents.


## Installation


``` pip install numpy matplotlib scikit-learn tensorflow ```


## Original Task Description

Implement a model that predicts whether an incident will occur within the next H time steps based on the previous W steps of one or more time-series metrics. Use a sliding-window formulation and train the model using any standard machine-learning framework.

The applicant may use any suitable public dataset or generate a synthetic time series with labeled incident intervals (e.g. anomalies or threshold breaches). The emphasis is on correct problem formulation, model selection, training, and evaluation rather than dataset complexity or model size.

The solution should include a clear description of the modeling choices, the evaluation setup (including alert thresholds and metrics), and an analysis of the results. During follow-up, the applicant should be able to explain the design decisions, discuss limitations, and outline how the approach could be adapted to a real alerting system.
