# AI-Powered IoT Monitoring Dashboard

A real-time machine monitoring dashboard using MQTT, Bokeh, and a TensorFlow model to predict equipment failure probabilities.

## Features
- Live data stream via MQTT
- Failure prediction with a trained neural network
- Task scheduling and worker assignment UI
- Editable task table with Bokeh widgets

## How to Run

1. Start the MQTT simulator:
    ```bash
    python mqtt_simulator.py
    ```

2. Start the dashboard (using Bokeh server):
    ```bash
    bokeh serve --show dashboard.py
    ```

## Dependencies
See `requirements.txt`.
