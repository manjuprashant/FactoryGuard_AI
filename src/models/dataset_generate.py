import pandas as pd
import random

texts = []
labels = []

safe_samples = [
    "All workers wearing helmets",
    "Machine operating normally",
    "No safety violations detected",
    "Fire extinguisher accessible",
    "Safety inspection completed"
]

unsafe_samples = [
    "Worker not wearing helmet",
    "Oil spill on floor",
    "Machine overheating",
    "Blocked emergency exit",
    "Electrical sparks detected"
]

for _ in range(150):
    texts.append(random.choice(safe_samples))
    labels.append(0)

for _ in range(150):
    texts.append(random.choice(unsafe_samples))
    labels.append(1)

df = pd.DataFrame({"text": texts, "label": labels})
df.to_csv("data/factory_guard_data.csv", index=False)
print("✅ Dataset saved to data/factory_guard_data.csv")
