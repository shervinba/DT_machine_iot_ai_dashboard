import json, time, numpy as np
import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
TOPIC = "factory/machine/data"

client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

while True:
    data = {
        "vibration": np.random.uniform(0.5, 2.5),
        "temperature": np.random.uniform(30, 80)
    }
    client.publish(TOPIC, json.dumps(data))
    print("Published:", data)
    time.sleep(5)
