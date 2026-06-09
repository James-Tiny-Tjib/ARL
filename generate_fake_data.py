import csv
import random
import time
import numpy as np

NUM_SENSORS = 23
NUM_ROWS = 500
OUTPUT_PATH = "./snapshot_log.csv"

def generate_row(label):
    is_threat = label == 1
    # Higher confidence when threat, lower when not
    base_conf = np.random.uniform(0.5, 1.0) if is_threat else np.random.uniform(0.0, 0.4)
    noise = lambda: np.random.uniform(-0.1, 0.1)

    rf_conf = float(np.clip(base_conf * 100 + noise() * 100, 0, 100))
    audio_mambo = float(np.clip(base_conf * 100 + noise() * 100, 0, 100)) if is_threat else float(np.random.uniform(0, 20))
    audio_bebop = float(np.clip(base_conf * 80 + noise() * 100, 0, 100)) if is_threat else float(np.random.uniform(0, 20))
    audio_background = float(np.clip(100 - audio_mambo - audio_bebop, 0, 100))
    visual_conf = float(np.clip(base_conf * 100 + noise() * 100, 0, 100)) if is_threat else float(np.random.uniform(0, 40))

    nearest = random.randint(0, 22)
    nearest_dist = float(np.random.uniform(5, 80) if is_threat else np.random.uniform(50, 500))

    sensor_threat = []
    sensor_friendly = []
    for _ in range(NUM_SENSORS):
        t = float(np.clip(base_conf + noise(), 0, 1)) if is_threat else float(np.clip(np.random.uniform(0, 0.3) + noise(), 0, 1))
        f = float(np.clip(1 - t + noise(), 0, 1))
        sensor_threat.append(t)
        sensor_friendly.append(f)

    row = {
        "timestamp": time.time() + random.uniform(-3600, 0),
        "nearest_sensor": nearest,
        "nearest_sensor_dist": nearest_dist,
        "rf_confidence": rf_conf,
        "rf_is_threat": is_threat,
        "audio_mambo": audio_mambo,
        "audio_bebop": audio_bebop,
        "audio_background": audio_background,
        "audio_is_threat": (audio_mambo + audio_bebop) > 70,
        "visual_confidence": visual_conf,
        "visual_is_threat": visual_conf >= 50,
        "label": label,
    }
    for i in range(NUM_SENSORS):
        row[f"sensor_{i}_threat_conf"] = sensor_threat[i]
        row[f"sensor_{i}_friendly_conf"] = sensor_friendly[i]
    return row

rows = [generate_row(1 if i < 250 else 0) for i in range(NUM_ROWS)]
random.shuffle(rows)

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {NUM_ROWS} rows to {OUTPUT_PATH}")
print(f"Threat rows: {sum(r['label'] for r in rows)}, Clear rows: {sum(1-r['label'] for r in rows)}")
