import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.font_manager import FontProperties

# Set the style
plt.style.use('bmh')  # Using a built-in matplotlib style
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.figsize'] = [20, 16]

# Read the CSV file from data folder
df = pd.read_csv('Data/Model_Evaluation_Summary.csv')

# Create a figure with 2x2 subplots
fig = plt.figure()

# 1. Model Performance Comparison
plt.subplot(2, 2, 1, polar=True)
performance_metrics = ['Uniqueness (%)', 'Group Match (%)', 'Reactive (%)']
x = np.arange(len(df['Model']))
width = 0.25
angles = np.linspace(0, 2*np.pi, len(performance_metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # complete the circle

for idx, model in enumerate(df['Model']):
    values = df.loc[idx, performance_metrics].values
    values = np.concatenate((values, [values[0]]))  # complete the circle
    plt.plot(angles, values, 'o-', linewidth=2, label=model)
    plt.fill(angles, values, alpha=0.25)

plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in performance_metrics])
plt.title('A)Model Performance Radar Chart', fontweight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))

# for i, metric in enumerate(performance_metrics):
#     plt.bar(x + i*width, df[metric], width, label=metric.replace('_', ' ').title())

# plt.xlabel('Models', fontweight='bold')
# plt.ylabel('Score', fontweight='bold')
# plt.title('Model Performance Comparison', fontweight='bold', pad=20)
# plt.xticks(x + width, df['Model'], rotation=45)
# plt.legend()

# 2. Sampling Distribution
# plt.subplot(2, 2, 2)
# sampling_metrics = ['Total Response', '#Unique Generated Valid Samples', 'Valid Samples']
# x = np.arange(len(df['Model']))

# for i, metric in enumerate(sampling_metrics):
#     plt.bar(x + i*width, df[metric], width, label=metric.replace('_', ' ').title())

# plt.xlabel('Models', fontweight='bold')
# plt.ylabel('Count', fontweight='bold')
# plt.title('B)Sampling Distribution', fontweight='bold', pad=20)
# plt.xticks(x + width, df['Model'], rotation=45)
# plt.legend()

width = 0.25
sampling_metrics = ['Total Response', '#Unique Generated Valid Samples', 'Valid Samples']
x = np.arange(len(df['Model']))

plt.subplot(2, 2, 2)

# Create bars and store the bar containers to control legend manually
bars = []
labels = ['Total Samples', 'Unique Valid Samples', 'Valid Samples']  # Custom legend labels

for i, metric in enumerate(sampling_metrics):
    bar = plt.bar(x + i*width, df[metric], width)
    bars.append(bar)

plt.xlabel('Models', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('B) Sampling Distribution', fontweight='bold', pad=20)
plt.xticks(x + width, df['Model'], rotation=45)

# Custom legend
plt.legend([bar[0] for bar in bars], labels, loc='upper right', fontsize='small', frameon=True)

# 3. Model Performance Heatmap
# plt.subplot(2, 2, 3)
# performance_data = df[performance_metrics].values
# sns.heatmap(performance_data, 
#             annot=True, 
#             fmt='.2f',
#             xticklabels=[m.replace('_', ' ').title() for m in performance_metrics],
#             yticklabels=df['Model'],
#             cmap='YlOrRd')
# plt.title('Model Performance Heatmap', fontweight='bold', pad=20)

# 4. Radar Chart for Model Performance
# plt.subplot(2, 2, 4, polar=True)
# angles = np.linspace(0, 2*np.pi, len(performance_metrics), endpoint=False)
# angles = np.concatenate((angles, [angles[0]]))  # complete the circle

# for idx, model in enumerate(df['Model']):
#     values = df.loc[idx, performance_metrics].values
#     values = np.concatenate((values, [values[0]]))  # complete the circle
#     plt.plot(angles, values, 'o-', linewidth=2, label=model)
#     plt.fill(angles, values, alpha=0.25)

# plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in performance_metrics])
# plt.title('Model Performance Radar Chart', fontweight='bold', pad=20)
# plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))

# Adjust layout and save
plt.tight_layout()
plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('model_performance_analysis.pdf', bbox_inches='tight')
plt.close()

print("Plots have been saved as 'model_performance_analysis.png' and 'model_performance_analysis.pdf'")


