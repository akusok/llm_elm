# data was saved as: pd.DataFrame(experimental_results).to_pickle(fname_out)

import os
import pandas as pd

experiments_dir = './experiments'
pkl_files = []

# Use os.walk to find all pkl files
for root, dirs, files in os.walk(experiments_dir):
    for file in files:
        if file.endswith('.pkl'):
            pkl_files.append(os.path.join(root, file))

if pkl_files:
    print(f"Found {len(pkl_files)} pickle files")
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each pickle file into a DataFrame and collect them
    for pkl_file in pkl_files:
        print(f"Reading: {pkl_file}")
        df = pd.read_pickle(pkl_file)
        dfs.append(df)
        print(f"  Shape: {df.shape}")
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Display information about the merged DataFrame
    print(f"\nMerged DataFrame shape: {merged_df.shape}")
    print("\nMerged DataFrame preview:")
    print(merged_df.head())
    
    # Save the merged DataFrame
    output_path = "./experiments.pkl"
    merged_df.to_pickle(output_path)
    print(f"\nMerged DataFrame saved to: {output_path}")
else:
    print("No pickle files found in the experiments directory.")