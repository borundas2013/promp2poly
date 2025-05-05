import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def create_model_comparison_plots():
    # Data preparation
    models = ['GPT-4o-mini', 'Mistral', 'DeepSeek', 'Llama 3.2']
    uniqueness = [14.52, 12.97, 22.32, 28.28]
    group_match = [61.00, 35.26, 40.00, 27.08]
    reactive = [42.00, 23.72, 22.81, 28.88]
    avg_per_query = [9.72, 6.16, 9.72, 6.16]
    total_samples = [1205, 1835, 1738, 1393]
    unique_samples = [175, 238, 388, 394]

    # Set style and font
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Add main title with padding
    plt.suptitle('Model Performance Analysis', fontsize=24, y=1.02, weight='bold')
    
    # Define vibrant color palette
    colors = {
        'uniqueness': '#FF6B6B',    # Coral Red
        'group_match': '#4ECDC4',   # Turquoise
        'reactive': '#45B7D1',      # Sky Blue
        'total': '#96CEB4',         # Sage Green
        'unique': '#FFEEAD',        # Light Yellow
        'reactive_samples': '#D4A5A5', # Dusty Rose
        'line': '#9B59B6'          # Purple
    }
    
    # 1. Grouped Bar Plot (top-left)
    ax1 = fig.add_subplot(221)
    x = np.arange(len(models))
    width = 0.25

    ax1.bar(x - width, uniqueness, width, label='Uniqueness (%)', color=colors['uniqueness'])
    ax1.bar(x, group_match, width, label='Group Matching (%)', color=colors['group_match'])
    ax1.bar(x + width, reactive, width, label='Reactive Success (%)', color=colors['reactive'])

    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.set_title('A) Model Performance Comparison', pad=20, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, fontweight='bold')
    ax1.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)

    # 2. Stacked Bar Plot (top-right)
    ax2 = fig.add_subplot(222)
    df = pd.DataFrame({
        'Model': models,
        'Total': total_samples,
        'Unique': unique_samples,
        'Reactive': [int(r * u / 100) for r, u in zip(reactive, unique_samples)]
    })

    df.plot(kind='bar', x='Model', y=['Total', 'Unique', 'Reactive'], 
            stacked=True, ax=ax2, color=[colors['total'], colors['unique'], colors['reactive_samples']])
    ax2.set_title('B) Sample Distribution Across Models', pad=20, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_xticklabels(models, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Scatter Plot (bottom-left)
    ax3 = fig.add_subplot(223)
    scatter = ax3.scatter(uniqueness, reactive, s=[g*10 for g in group_match], 
                         alpha=0.8, c=group_match, cmap='viridis')
    
    for i, model in enumerate(models):
        ax3.annotate(model, (uniqueness[i], reactive[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=12)

    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Group Matching (%)', fontweight='bold', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax3.set_xlabel('Uniqueness (%)', fontweight='bold')
    ax3.set_ylabel('Reactive Success (%)', fontweight='bold')
    ax3.set_title('C) Uniqueness vs Reactivity Trade-off', pad=20, fontweight='bold')

    # 4. Line Chart (bottom-right)
    ax4 = fig.add_subplot(224)
    ax4.plot(models, avg_per_query, marker='o', linewidth=2, markersize=10, 
             color=colors['line'], linestyle='--')
    ax4.set_ylabel('Average Unique Pairs per Query', fontweight='bold')
    ax4.set_title('D) Model Efficiency: Unique Outputs per Query', pad=20, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_xticklabels(models, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figures with extra padding at the top for the title
    plt.savefig('model_analysis.pdf', format='pdf', bbox_inches='tight', pad_inches=0.5)
    plt.savefig('model_analysis.png', dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.close()

if __name__ == "__main__":
    create_model_comparison_plots()