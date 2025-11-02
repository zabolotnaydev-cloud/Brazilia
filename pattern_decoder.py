import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and combine all data
all_files = [f'dataset/dataset_part_{i}.csv' for i in range(1, 6)]
df = pd.concat([pd.read_csv(f) for f in all_files])

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Method 1: Plot only zero-noise data, separate by series
print("Creating visualizations...")

# Filter for clean signals
clean_df = df[df['noise_level'] == 0.0].sort_values('timestamp')

print(f"Total rows: {len(df)}")
print(f"Clean rows (noise=0): {len(clean_df)}")
print(f"Unique series: {clean_df['series_id'].unique()}")

# Create multiple visualizations
fig, axes = plt.subplots(3, 2, figsize=(20, 15))

# 1. Scatter plot - all clean points colored by series
ax = axes[0, 0]
for series_id in clean_df['series_id'].unique():
    series_data = clean_df[clean_df['series_id'] == series_id]
    ax.scatter(series_data['timestamp'], series_data['value'], 
               label=series_id, s=50, alpha=0.7)
ax.set_title('Scatter: Clean Data by Series', fontsize=14)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# 2. Connected lines - each series separately
ax = axes[0, 1]
colors = plt.cm.tab10.colors
for idx, series_id in enumerate(sorted(clean_df['series_id'].unique())):
    series_data = clean_df[clean_df['series_id'] == series_id].sort_values('timestamp')
    ax.plot(series_data['timestamp'], series_data['value'], 
            marker='o', linewidth=2, markersize=6, 
            label=series_id, color=colors[idx % len(colors)])
ax.set_title('Connected Lines: Each Series', fontsize=14)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Value')
ax.legend()
ax.grid(alpha=0.3)

# 3. Stacked view - each series in separate row
ax = axes[1, 0]
series_ids = sorted(clean_df['series_id'].unique())
for idx, series_id in enumerate(series_ids):
    series_data = clean_df[clean_df['series_id'] == series_id].sort_values('timestamp')
    # Offset each series vertically
    ax.plot(series_data['timestamp'], series_data['value'] + idx*12, 
            marker='o', linewidth=2, markersize=4, label=series_id)
ax.set_title('Stacked Series View', fontsize=14)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Value (offset)')
ax.legend()
ax.grid(alpha=0.3)

# 4. Normalize time to sequential index
ax = axes[1, 1]
clean_df_indexed = clean_df.copy().reset_index(drop=True)
clean_df_indexed['seq_index'] = range(len(clean_df_indexed))
ax.plot(clean_df_indexed['seq_index'], clean_df_indexed['value'], 
        marker='o', linewidth=2, markersize=6, color='purple')
ax.set_title('Sequential Index View (ignoring time gaps)', fontsize=14)
ax.set_xlabel('Sequential Index')
ax.set_ylabel('Value')
ax.grid(alpha=0.3)

# 5. Value over time - highlight patterns with stem plot
ax = axes[2, 0]
ax.stem(clean_df['timestamp'], clean_df['value'], basefmt=' ')
ax.set_title('Stem Plot: Clean Signal Patterns', fontsize=14)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Value')
ax.grid(alpha=0.3)

# 6. Grouped by time windows to see letter patterns
ax = axes[2, 1]
# Group data into time windows
clean_df['time_window'] = pd.cut(clean_df['timestamp'], bins=20)
for window in clean_df['time_window'].unique():
    window_data = clean_df[clean_df['time_window'] == window].sort_values('value')
    if len(window_data) > 0:
        # Use window midpoint as x-coordinate
        x_pos = window_data['timestamp'].mean()
        ax.scatter([x_pos] * len(window_data), window_data['value'], 
                   s=100, alpha=0.6)
        # Connect points vertically
        ax.plot([x_pos] * len(window_data), window_data['value'], 
                linewidth=3, alpha=0.5)
ax.set_title('Vertical Letter Pattern Detection', fontsize=14)
ax.set_xlabel('Time Window')
ax.set_ylabel('Value')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pattern_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: pattern_analysis.png")
plt.close()

# Additional analysis: Look at value distribution
print("\nValue statistics:")
print(clean_df['value'].describe())
print(f"\nValue range: {clean_df['value'].min():.2f} to {clean_df['value'].max():.2f}")

# Check if values form discrete levels (like letter strokes)
unique_values = np.round(clean_df['value'], 1)
print(f"\nMost common value levels:")
print(unique_values.value_counts().head)
