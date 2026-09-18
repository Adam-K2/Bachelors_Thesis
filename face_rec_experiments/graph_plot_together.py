# Script to plot probability density function in one graph

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('results_l2.csv')
#df = pd.read_csv('results_cos.csv')

# Avoid rows without values
df = df.dropna()

# Plot function
def plot_distribution_for_model(df, model_name):
    plt.figure(figsize=(8, 6))
    
    sns.kdeplot(df[f'{model_name}_genuine'], label=f'{model_name} Genuine', shade=True, bw_adjust=0.5)
    sns.kdeplot(df[f'{model_name}_fake'], label=f'{model_name} Fake', shade=True, bw_adjust=0.5)
    
    plt.title(f'Probability Density Function of {model_name} Distances')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Use face recognition algorithms
models = ['FaceNet', 'DeepFace', 'Arcface', 'Magface']
#models = ['FaceNet', 'DeepFace', 'Arcface'] # For cos metric

for model in models:
    plot_distribution_for_model(df, model)
