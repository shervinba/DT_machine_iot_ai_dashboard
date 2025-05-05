import json
import threading
import queue
import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, DataTable, TableColumn, TextInput, Button, Select
from bokeh.plotting import figure, curdoc
import paho.mqtt.client as mqtt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Simulate dataset
data = {
    "vibration": np.random.uniform(0.5, 2.5, 1000),
    "temperature": np.random.uniform(30, 80, 1000),
    "failure": np.random.choice([0, 1], size=1000, p=[0.9, 0.1])
}
df = pd.DataFrame(data)

# Train model
X = df[['vibration', 'temperature']]
y = df['failure']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

# --- MQTT Setup ---
MQTT_BROKER = "broker.hivemq.com"
TOPIC = "factory/machine/data"
mqtt_queue = queue.Queue()

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with code:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        mqtt_queue.put(payload)
    except json.JSONDecodeError:
        print("Received invalid JSON")

def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, 1883, 60)
    client.loop_forever()

threading.Thread(target=mqtt_thread, daemon=True).start()

# --- Worker and Task Management ---
workers = [
    {"id": i, "name": f"Worker {chr(65+i)}", "available": True} for i in range(20)
]
tasks = []
task_id_counter = 1

def assign_worker(probability):
    global task_id_counter
    available_workers = [w for w in workers if w["available"]]
    if not available_workers:
        return None, "No workers available"
    
    worker = available_workers[0]
    worker["available"] = False
    
    due_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    
    task = {
        "task_id": task_id_counter,
        "priority": round(probability, 2),
        "worker": worker["name"],
        "status": "Scheduled",
        "due_date": due_date
    }
    tasks.append(task)
    task_id_counter += 1
    return task, worker["name"]

# --- Bokeh Setup ---
source = ColumnDataSource(data=dict(x=[], vibration=[], temperature=[], failure_prob=[]))

# Plots
plot1 = figure(title="Vibration", x_axis_label="Time", y_axis_label="Vibration", height=300, width=600)
plot1.line(source=source, x='x', y='vibration', line_width=2, color="blue")

plot2 = figure(title="Temperature", x_axis_label="Time", y_axis_label="Temp (°F)", height=300, width=600)
plot2.line(source=source, x='x', y='temperature', line_width=2, color="red")

plot3 = figure(title="Failure Probability", x_axis_label="Time", y_axis_label="Probability", height=300, width=600)
plot3.line(source=source, x='x', y='failure_prob', line_width=2, color="purple")

# Status Div
status_div = Div(text="<span style='color:green;'>✅ Waiting for data...</span>", width=600)

# Maintenance Schedule Table
task_source = ColumnDataSource(data=dict(task_id=[], priority=[], worker=[], status=[], due_date=[]))
columns = [
    TableColumn(field="task_id", title="Task ID"),
    TableColumn(field="priority", title="Priority (Prob)"),
    TableColumn(field="worker", title="Assigned Worker"),
    TableColumn(field="status", title="Status"),
    TableColumn(field="due_date", title="Due Date")
]
task_table = DataTable(source=task_source, columns=columns, width=500, height=300, editable=True)
table_title = Div(text="<b>Maintenance Schedule</b>", width=500, styles={'font-size': '16px', 'margin-bottom': '10px'})

# Editing Widgets
task_id_input = TextInput(title="Select Task ID:", value="")
priority_input = TextInput(title="Edit Priority (0-1):", value="0.0")
worker_select = Select(title="Reassign Worker:", options=[w["name"] for w in workers if w["available"]] + [""])
due_date_input = TextInput(title="Edit Due Date (YYYY-MM-DD):", value="")
status_select = Select(title="Edit Status:", options=["Scheduled", "In Progress", "Completed"], value="Scheduled")
update_button = Button(label="Update Task", button_type="success")

# Track selected task
selected_task_id = None

def select_task_by_id():
    global selected_task_id
    try:
        task_id = int(task_id_input.value)
        if task_id not in task_source.data["task_id"]:
            status_div.text = f"<span style='color:red;'>⚠️ Task ID {task_id} not found</span>"
            selected_task_id = None
            task_table.source.selected.indices = []
            priority_input.value = "0.0"
            worker_select.value = ""
            due_date_input.value = ""
            status_select.value = "Scheduled"
            return
        
        selected_task_id = task_id
        task = next(t for t in tasks if t["task_id"] == task_id)
        priority_input.value = str(task["priority"])
        worker_select.value = task["worker"]
        due_date_input.value = task["due_date"]
        status_select.value = task["status"]
        
        # Highlight the selected row in the table
        task_index = task_source.data["task_id"].index(task_id)
        task_table.source.selected.indices = [task_index]
        status_div.text = f"<span style='color:green;'>✅ Selected Task ID {task_id}</span>"
        
        # Update worker select options
        worker_select.options = [w["name"] for w in workers if w["available"] or w["name"] == task["worker"]] + [""]
    except ValueError:
        status_div.text = "<span style='color:red;'>⚠️ Invalid Task ID</span>"
        selected_task_id = None
        task_table.source.selected.indices = []

def on_table_select(attr, old, new):
    global selected_task_id
    if new:
        selected_task_id = task_source.data["task_id"][new[0]]
        task = next(t for t in tasks if t["task_id"] == selected_task_id)
        task_id_input.value = str(selected_task_id)
        priority_input.value = str(task["priority"])
        worker_select.value = task["worker"]
        due_date_input.value = task["due_date"]
        status_select.value = task["status"]
        worker_select.options = [w["name"] for w in workers if w["available"] or w["name"] == task["worker"]] + [""]
        status_div.text = f"<span style='color:green;'>✅ Selected Task ID {selected_task_id}</span>"
    else:
        selected_task_id = None
        task_id_input.value = ""
        priority_input.value = "0.0"
        worker_select.value = ""
        due_date_input.value = ""
        status_select.value = "Scheduled"

task_id_input.on_change('value', lambda attr, old, new: select_task_by_id())
task_table.source.selected.on_change('indices', on_table_select)

def update_task():
    global selected_task_id
    if selected_task_id is None:
        status_div.text = "<span style='color:orange;'>⚠️ Select a task to update</span>"
        return
    
    try:
        new_priority = float(priority_input.value)
        if not 0 <= new_priority <= 1:
            raise ValueError("Priority must be between 0 and 1")
        
        new_worker = worker_select.value
        new_due_date = due_date_input.value
        new_status = status_select.value
        
        # Validate due date format
        datetime.strptime(new_due_date, "%Y-%m-%d")
        
        # Update task in tasks list
        task = next(t for t in tasks if t["task_id"] == selected_task_id)
        old_worker = task["worker"]
        
        # Update worker availability
        if old_worker != new_worker:
            old_worker_obj = next(w for w in workers if w["name"] == old_worker)
            old_worker_obj["available"] = True
            if new_worker:
                new_worker_obj = next(w for w in workers if w["name"] == new_worker)
                new_worker_obj["available"] = False
        
        # Update task
        task["priority"] = new_priority
        task["worker"] = new_worker
        task["due_date"] = new_due_date
        task["status"] = new_status
        
        # Update task_source
        task_index = task_source.data["task_id"].index(selected_task_id)
        task_source.data["priority"][task_index] = new_priority
        task_source.data["worker"][task_index] = new_worker
        task_source.data["due_date"][task_index] = new_due_date
        task_source.data["status"][task_index] = new_status
        task_source.data = dict(task_source.data)  # Trigger update
        
        # Update priority_source
        priority_index = [i for i, tid in enumerate(priority_source.data["task_id"]) if tid == selected_task_id]
        if priority_index:
            priority_source.data["priority"][priority_index[0]] = new_priority
            priority_source.data = dict(priority_source.data)
        
        # Update worker select options
        worker_select.options = [w["name"] for w in workers if w["available"] or w["name"] == new_worker] + [""]
        
        status_div.text = f"<span style='color:green;'>✅ Task ID {selected_task_id} updated successfully</span>"
    except ValueError as e:
        status_div.text = f"<span style='color:red;'>⚠️ Error: {str(e)}</span>"

update_button.on_click(update_task)

# Canvas Panel: Priority Visualization
priority_plot = figure(title="Task Priorities", x_axis_label="Task ID", y_axis_label="Priority", height=300, width=500)
priority_source = ColumnDataSource(data=dict(task_id=[], priority=[]))
priority_plot.scatter(source=priority_source, x='task_id', y='priority', size=10, color="purple")

def update():
    while not mqtt_queue.empty():
        data = mqtt_queue.get()
        vibration = data.get("vibration", np.nan)
        temperature = data.get("temperature", np.nan)

        sample = np.array([[vibration, temperature]])
        sample_scaled = scaler.transform(sample)
        prediction = float(model.predict(sample_scaled, verbose=0)[0][0])

        new_data = {
            "x": [len(source.data["x"])],
            "vibration": [vibration],
            "temperature": [temperature],
            "failure_prob": [prediction]
        }
        source.stream(new_data, rollover=100)

        if prediction > 0.1:
            status_div.text = f"<b style='color:red;'>⚠️ ALERT: High risk of failure ({prediction:.2f})</b>"
            task, worker_name = assign_worker(prediction)
            if task:
                task_source.stream({
                    "task_id": [task["task_id"]],
                    "priority": [task["priority"]],
                    "worker": [task["worker"]],
                    "status": [task["status"]],
                    "due_date": [task["due_date"]]
                })
                priority_source.stream({
                    "task_id": [task["task_id"]],
                    "priority": [task["priority"]]
                })
                worker_select.options = [w["name"] for w in workers if w["available"] or w["name"] == task["worker"]] + [""]
            else:
                status_div.text += f"<br><span style='color:orange;'>⚠️ {worker_name}</span>"
        else:
            status_div.text = f"<span style='color:green;'>✅ Normal operation ({prediction:.2f})</span>"

# Layout
left_column = column(plot1, plot2, plot3, status_div)
right_column = column(
    table_title, task_table, priority_plot,
    task_id_input, priority_input, worker_select, due_date_input, status_select, update_button
)
layout = row(left_column, right_column)
curdoc().add_root(layout)
curdoc().add_periodic_callback(update, 1000)