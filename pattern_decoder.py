import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

# Load and combine all data
all_files = [f'dataset/dataset_part_{i}.csv' for i in range(1,6)]
df = pd.concat([pd.read_csv(f) for f in all_files])

# Convert timestamp and filter clean signals
df['timestamp'] = pd.to_datetime(df['timestamp'])
clean_df = df[df['noise_level'] == 0.0].sort_values('timestamp')

# Create plotting canvas
fig, ax = plt.subplots(figsize=(20, 6))

# Plot each clean data point as segments
prev_point = None
prev_series = None

for _, row in clean_df.iterrows():
    if prev_point and row['series_id'] == prev_series:
        # Connect points from same series
        ax.plot([prev_point[0], row['timestamp']], 
                [prev_point[1], row['value']],
                color='black', linewidth=2)
    prev_point = (row['timestamp'], row['value'])
    prev_series = row['series_id']

# Format plot
ax.set_title('Connected Clean Signal Points', fontsize=16)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('connected_pattern.png', dpi=300)
plt.close()

print("New visualization saved as connected_pattern.png")