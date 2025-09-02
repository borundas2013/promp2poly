import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv_file(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print(df.columns)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def create_mae_bar_plot(df):
    # Filter out rows with empty Models and valid MAE values
    valid_data = df[df['Models'].notna() & df['Models'].str.strip() != ''].copy()
    
    # Clean up the data - remove any rows with NaN MAE values
    valid_data = valid_data.dropna(subset=['Diff Tg (MAE)', 'Diff Er (MAE)'])
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the width of each bar
    bar_width = 0.35
    
    # Set the positions of the bars
    x = range(len(valid_data))
    
    # Create bars for Tg MAE with deeper colors
    bars1 = ax.bar([i - bar_width/2 for i in x], valid_data['Diff Tg (MAE)'], 
                    bar_width, label='Glass Transition Temperature (T$_g$)', color='darkblue', alpha=0.8)
    
    # Create bars for Er MAE with deeper colors
    bars2 = ax.bar([i + bar_width/2 for i in x], valid_data['Diff Er (MAE)'], 
                    bar_width, label='Recovery Stress (E$_r$)', color='darkred', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Models', fontsize=16, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=16, fontweight='bold')
    ax.set_title('Comparison of T$_g$ and E$_r$ MAE Values Across Models', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_data['Models'], rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14, prop={'weight': 'bold'})
    ax.grid(axis='y', alpha=0.3)
    
    # Make y-axis tick labels bold and bigger
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot as SVG
    plt.savefig('mae_comparison_plot.svg', format='svg', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    return valid_data


def find_duplicate_samples(gpt_df, deepseek_df, llama_df):
    """
    Find and analyze duplicate samples across all models
    """
    results = {}
    
    for model_name, df in [('GPT4', gpt_df), ('DeepSeek', deepseek_df), ('Llama32', llama_df)]:
        if df is None or df.empty:
            continue
            
        # Filter out rows with 'Not found' monomers
        filtered_df = df[(df['Fixed Monomer 1'] != 'Not found') & (df['Fixed Monomer 2'] != 'Not found')]
        
        if filtered_df.empty:
            continue
        
        # Find duplicates based on monomer pairs
        duplicates = filtered_df[filtered_df.duplicated(subset=['Fixed Monomer 1', 'Fixed Monomer 2'], keep=False)]
        
        # Group duplicates by monomer pair
        duplicate_groups = {}
        for _, row in duplicates.iterrows():
            pair = (row['Fixed Monomer 1'], row['Fixed Monomer 2'])
            if pair not in duplicate_groups:
                duplicate_groups[pair] = []
            duplicate_groups[pair].append({
                'Temperature': row['Temperature'],
                'Index': row.name,
                'Row_Data': row.to_dict()
            })
        
        # Filter only groups with more than 1 occurrence (actual duplicates)
        actual_duplicates = {pair: data for pair, data in duplicate_groups.items() if len(data) > 1}
        
        results[model_name] = {
            'total_samples': len(filtered_df),
            'unique_samples': len(filtered_df.drop_duplicates(subset=['Fixed Monomer 1', 'Fixed Monomer 2'])),
            'duplicate_samples': len(duplicates),
            'duplicate_pairs': actual_duplicates,
            'duplicate_count': len(actual_duplicates)
        }
    
    return results

def draw_sampling_temperature_plot(gpt_df, deepseek_df, llama_df):
    gpt_df= gpt_df[(gpt_df['Fixed Monomer 1']!='Not found') & (gpt_df['Fixed Monomer 2']!='Not found')]
    deepseek_df= deepseek_df[(deepseek_df['Fixed Monomer 1']!='Not found') & (deepseek_df['Fixed Monomer 2']!='Not found')]
    llama_df= llama_df[(llama_df['Fixed Monomer 1']!='Not found') & (llama_df['Fixed Monomer 2']!='Not found')]

    # unique_gpt_df = gpt_df.drop_duplicates(subset=['Fixed Monomer 1', 'Fixed Monomer 2'])
    # unique_deepseek_df = deepseek_df.drop_duplicates(subset=['Fixed Monomer 1', 'Fixed Monomer 2'])
    # unique_llama_df = llama_df.drop_duplicates(subset=['Fixed Monomer 1', 'Fixed Monomer 2'])

    # print(f"Unique samples - GPT4: {len(unique_gpt_df)}, DeepSeek: {len(unique_deepseek_df)}, Llama32: {len(unique_llama_df)}")

    # Find and analyze duplicates
    duplicate_analysis = find_duplicate_samples(gpt_df, deepseek_df, llama_df)
    
    # Print duplicate analysis
    print("\n=== DUPLICATE SAMPLE ANALYSIS ===")
    for model_name, data in duplicate_analysis.items():
        print(f"\n{model_name}:")
        print(f"  Total samples: {data['total_samples']}")
        print(f"  Unique samples: {data['unique_samples']}")
        print(f"  Duplicate samples: {data['duplicate_samples']}")
        print(f"  Duplicate pairs: {data['duplicate_count']}")
        
        if data['duplicate_pairs']:
            print(f"  Duplicate details:")
            for pair, occurrences in list(data['duplicate_pairs'].items())[:5]:  # Show first 5 duplicates
                print(f"    Monomer pair: {pair[0]} + {pair[1]}")
                print(f"    Occurrences: {len(occurrences)}")
                temps = [occ['Temperature'] for occ in occurrences]
                print(f"    Temperatures: {temps}")
                print()

    
    
    # Create temperature vs uniqueness plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Temperature vs Sample Uniqueness Analysis', fontsize=20, fontweight='bold')
    
    # Store legend handles and labels from first subplot
    legend_handles = []
    legend_labels = []
    
    for idx, (model_name, df) in enumerate([('GPT4', gpt_df), ('DeepSeek', deepseek_df), ('Llama32', llama_df)]):
        if df is None or df.empty:
            continue
            
        ax = axes[idx]
        temp_stats = []
        
        for temp in sorted(df['Temperature'].unique()):
            temp_data = df[df['Temperature'] == temp]
            total = len(temp_data)
            unique = len(temp_data.drop_duplicates(subset=['Fixed Monomer 1', 'Fixed Monomer 2']))
            duplicates = total - unique
            
            temp_stats.append({
                'temp': temp,
                'total': total,
                'unique': unique,
                'duplicates': duplicates,
                'uniqueness_ratio': unique/total if total > 0 else 0
            })
        
        if temp_stats:
            temps = [stat['temp'] for stat in temp_stats]
            unique_counts = [stat['unique'] for stat in temp_stats]
            duplicate_counts = [stat['duplicates'] for stat in temp_stats]
            
            x = range(len(temps))
            width = 0.3
            
            bars1 = ax.bar([i - width/2 for i in x], unique_counts, width, label='Unique', color='green', alpha=0.7)
            bars2 = ax.bar([i + width/2 for i in x], duplicate_counts, width, label='Duplicates', color='red', alpha=0.7)
            
            ax.set_xlabel('Temperature', fontsize=16, fontweight='bold')
            ax.set_ylabel('Count', fontsize=16, fontweight='bold')
            ax.set_title(f'{model_name}', fontsize=18, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(temps, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Make y-axis tick labels bold and bigger
            ax.tick_params(axis='y', labelsize=14)
            # Make y-axis labels bold using a different approach
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # Store legend handles and labels from first subplot only
            if idx == 0:
                legend_handles = [bars1, bars2]
                legend_labels = ['Unique', 'Duplicates']
    
    # Add single legend for all plots at the bottom (only if we have handles)
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                   ncol=2, fontsize=14, prop={'weight': 'bold'})
        # Adjust bottom margin to make room for legend
        plt.subplots_adjust(wspace=0.3, bottom=0.2)
    else:
        # If no legend, just adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_duplicate_analysis.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return duplicate_analysis


# Read the CSV file
df = read_csv_file('Result_Summary.csv')
#gpt_df = read_csv_file("GPT4/Output/Combined_gpt.csv")
##deepseek_df = read_csv_file("DeepSeek/Output/Combined_deepseek.csv")
#llama_df = read_csv_file("LLama32/Output/Combined_llama.csv")
if df is not None:
    valid_data = create_mae_bar_plot(df)
    print("\nData used for plotting:")
    print(valid_data[['Models', 'Diff Tg (MAE)', 'Diff Er (MAE)']])
else:
    print("Failed to read the CSV file.")

# Create the MAE bar plot
# if gpt_df is not None and deepseek_df is not None and llama_df is not None:
#     #valid_data = create_mae_bar_plot(df)
#     #draw_sampling_temperature_plot(gpt_df, deepseek_df, llama_df)
#     print("\nData used for plotting:")
#     #print(valid_data[['Models', 'Diff Tg (MAE)', 'Diff Er (MAE)']])
# else:
#     print("Failed to read the CSV file.")