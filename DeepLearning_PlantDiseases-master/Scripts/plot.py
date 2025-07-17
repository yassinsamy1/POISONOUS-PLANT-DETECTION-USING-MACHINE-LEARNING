import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

with open('stats.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    stats_full = list(reader)

# Convert all relevant fields to float for plotting
for s in stats_full:
    for key in ['f1_score', 'accuracy', 'precision', 'recall', 'training_time', 'max_loss', 'min_loss', 'avg_loss']:
        if key in s and s[key] != '':
            try:
                s[key] = float(s[key])
            except ValueError:
                s[key] = 0.0  # fallback if conversion fails

names = [str(s['name']) for s in stats_full]

# Only these metrics will be shown as bar charts
metrics = [
    ('f1_score', 'F1-Score'),
    ('accuracy', 'Accuracy'),
    ('precision', 'Precision'),
    ('recall', 'Recall')
]

# Plot selected metrics in a single figure with subplots (bar charts)
n_metrics = len(metrics)
fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))  # 1 row, n_metrics columns
if n_metrics == 1:
    axes = [axes]

for ax, (metric_key, metric_label) in zip(axes, metrics):
    values = [float(s[metric_key]) if metric_key in s else 0.0 for s in stats_full]
    bars = ax.bar(names, values, color=['blue', 'orange', 'green'][:len(names)])
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} Comparison')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha='center', va='bottom')
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.show()

# Show as a table (all metrics)
df = pd.DataFrame(stats_full)[['name', 'f1_score', 'accuracy', 'precision', 'recall', 'training_time', 'max_loss', 'min_loss', 'avg_loss']]
print("\nModel Performance Table (All Metrics):")
print(df.to_string(index=False))
