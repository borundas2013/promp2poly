import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
from Cheng_Model_Prediction import ChengModelPredictor

class PredictionAnalyzer:
    def __init__(self, reaction_file: str, non_reaction_file: str):
        """
        Initialize the PredictionAnalyzer with file paths
        
        Args:
            reaction_file (str): Path to the reaction pairs CSV file
            non_reaction_file (str): Path to the non-reaction pairs CSV file
        """
        self.reaction_file = reaction_file
        self.non_reaction_file = non_reaction_file
        self.reaction_data = None
        self.non_reaction_data = None
        
    def load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load the prediction data from CSV files
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing reaction and non-reaction dataframes
        """
        try:
            self.reaction_data = pd.read_csv(self.reaction_file)
            self.non_reaction_data = pd.read_csv(self.non_reaction_file)
            print("Reaction Data Sample:")
            print(self.reaction_data.head())
            print("\nNon-Reaction Data Sample:")
            print(self.non_reaction_data.head())
            return self.reaction_data, self.non_reaction_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def analyze_predictions(self, df: pd.DataFrame) -> None:
        """
        Analyze the predictions and generate basic statistics
        
        Args:
            df (pd.DataFrame): DataFrame containing prediction data
        """
        if df is None:
            return
        
        # Basic statistics
        print("\nBasic Statistics:")
        print("\nPredicted Er Statistics:")
        print(df['Predicted_Er'].describe())
        print("\nPredicted Tg Statistics:")
        print(df['Predicted_Tg'].describe())
        
        self._create_distribution_plots(df)
        self._analyze_correlations(df)
        self._analyze_molar_ratios(df)

    def _create_distribution_plots(self, df: pd.DataFrame) -> None:
        """
        Create distribution plots for Er and Tg predictions
        
        Args:
            df (pd.DataFrame): DataFrame containing prediction data
        """
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Distribution of Predicted Er
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x='Predicted_Er', bins=30)
        plt.title('Distribution of Predicted Er')
        plt.xlabel('Predicted Er')
        
        # Plot 2: Distribution of Predicted Tg
        plt.subplot(1, 2, 2)
        sns.histplot(data=df, x='Predicted_Tg', bins=30)
        plt.title('Distribution of Predicted Tg')
        plt.xlabel('Predicted Tg')
        
        plt.tight_layout()
        plt.savefig('prediction_distributions.png')
        plt.close()

    def _analyze_correlations(self, df: pd.DataFrame) -> None:
        """
        Analyze correlations between properties
        
        Args:
            df (pd.DataFrame): DataFrame containing prediction data
        """
        correlation = df[['Predicted_Er', 'Predicted_Tg']].corr()
        print("\nCorrelation between Er and Tg:")
        print(correlation)

    def _analyze_molar_ratios(self, df: pd.DataFrame) -> None:
        """
        Analyze molar ratio statistics
        
        Args:
            df (pd.DataFrame): DataFrame containing prediction data
        """
        print("\nMolar Ratio Statistics:")
        print(df[['Molar_Ratio_1', 'Molar_Ratio_2']].describe())

    def save_analysis(self, df: pd.DataFrame, output_file: str = 'analyzed_predictions.csv') -> None:
        """
        Save the analyzed data to a CSV file
        
        Args:
            df (pd.DataFrame): DataFrame containing prediction data
            output_file (str): Path to save the analyzed data
        """
        if df is not None:
            df.to_csv(output_file, index=False)
            print(f"\nAnalysis complete. Results saved to '{output_file}'")

    def compare_reaction_non_reaction(self) -> None:
        """
        Compare statistics between reaction and non-reaction pairs
        """
        if self.reaction_data is None or self.non_reaction_data is None:
            print("Please load data first using load_data()")
            return

        print("\nComparison between Reaction and Non-Reaction Pairs:")
        print("\nReaction Pairs Statistics:")
        print(self.reaction_data.describe())
        print("\nNon-Reaction Pairs Statistics:")
        print(self.non_reaction_data.describe())

def main():
    # Initialize the analyzer
    analyzer = PredictionAnalyzer(
        reaction_file='DeepSeek/Output/Analysis/all_reactive_pairs.csv',
        non_reaction_file='DeepSeek/Output/Analysis/all_non_reactive_pairs.csv'
    )
    
    # Load the data
    reaction_df, non_reaction_df = analyzer.load_data()

    # Define model paths
    model_paths = {
        'smiles_to_latent': "Yans_code/Blog_simple_smi2lat8_150",
        'latent_to_states': "Yans_code/Blog_simple_latstate8_150",
        'sample': "Yans_code/Blog_simple_samplemodel8_150"
    }
    
    # Define model weights
    model_weights = {
        'Tg': 'Yans_code/conv1d_model1_Tg245_3.h5',
        'Er': 'Yans_code/conv1d_model1_Er245_2.h5'
    }
    
    # Initialize predictor
    predictor = ChengModelPredictor(
        model_paths=model_paths,
        df=reaction_df
    )
    
    # Run predictions
    predictor.predict_properties(
        model_weights=model_weights,
        output_file='Yans_code/prediction_smiles_1.csv'
    )
    
    # # Analyze the data
    # if reaction_df is not None:
    #     analyzer.analyze_predictions(reaction_df)
    #     analyzer.save_analysis(reaction_df)
        
    # # Compare reaction and non-reaction pairs
    # analyzer.compare_reaction_non_reaction()

if __name__ == "__main__":
    main() 