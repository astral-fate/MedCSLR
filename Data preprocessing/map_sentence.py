import pandas as pd
import os

def merge_and_deduplicate(file1_path, file2_path, output_path):
    """
    Merges two CSV files, removes duplicate rows, and saves the result.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.
        output_path (str): Path to save the final merged and cleaned CSV file.
    """
    print(f"--- 🔄 Merging {os.path.basename(file1_path)} and {os.path.basename(file2_path)} ---")

    # --- 1. Check if input files exist ---
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print("❌ ERROR: One or both input files were not found.")
        if not os.path.exists(file1_path): print(f"  - Missing: {file1_path}")
        if not os.path.exists(file2_path): print(f"  - Missing: {file2_path}")
        return

    try:
        # --- 2. Load the datasets ---
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        print(f"✅ Loaded {len(df1)} rows from file 1 and {len(df2)} rows from file 2.")
        
        # --- 3. Combine the DataFrames ---
        combined_df = pd.concat([df1, df2], ignore_index=True)
        print(f"Total rows before deduplication: {len(combined_df)}")

        # --- 4. Remove duplicate rows ---
        deduplicated_df = combined_df.drop_duplicates()
        num_duplicates_removed = len(combined_df) - len(deduplicated_df)
        print(f"Removed {num_duplicates_removed} duplicate rows.")
        print(f"Total rows after deduplication: {len(deduplicated_df)}")

        # --- 5. Save the final dataset ---
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        deduplicated_df.to_csv(output_path, index=False)
        print(f"✅ Success! Saved final merged file to '{output_path}'")

    except Exception as e:
        print(f"An error occurred during the process: {e}")

def add_sentences_to_final_data(final_data_path, sentences_source_path, output_path):
    """
    Adds a 'sentence' column by mapping IDs and consolidates 'sentence_x'/'sentence_y' if they are created.

    Args:
        final_data_path (str): Path to the final merged data file (e.g., final_train.csv).
        sentences_source_path (str): Path to the master file with 'id' and 'sentence'.
        output_path (str): Path to save the new file with the added sentence column.
    """
    print(f"--- 📝 Adding sentences to {os.path.basename(final_data_path)} ---")

    # --- 1. Check if input files exist ---
    if not os.path.exists(final_data_path) or not os.path.exists(sentences_source_path):
        print("❌ ERROR: One or both input files were not found.")
        if not os.path.exists(final_data_path): print(f"  - Missing: {final_data_path}")
        if not os.path.exists(sentences_source_path): print(f"  - Missing: {sentences_source_path}")
        return None

    try:
        # --- 2. Load the datasets ---
        final_df = pd.read_csv(final_data_path)
        sentences_df = pd.read_csv(sentences_source_path)
        print(f"✅ Loaded {len(final_df)} rows from final data and {len(sentences_df)} rows from sentences source.")

        # --- 3. Merge to add the sentence column ---
        # This merge can create 'sentence_x' and 'sentence_y' if 'sentence' exists in both DataFrames.
        merged_df = pd.merge(final_df, sentences_df[['id', 'sentence']], on='id', how='left')
        
        # --- 4. NEW: Self-contained logic to consolidate sentence columns ---
        if 'sentence_y' in merged_df.columns and 'sentence_x' in merged_df.columns:
            print("Found 'sentence_x' and 'sentence_y'. Consolidating into a single 'sentence' column...")
            # Prioritize the sentence from the source file ('sentence_y').
            # If 'sentence_y' is null, fall back to the existing 'sentence_x'.
            merged_df['sentence'] = merged_df['sentence_y'].fillna(merged_df['sentence_x'])
            # Drop the old, separate columns
            merged_df.drop(columns=['sentence_x', 'sentence_y'], inplace=True)
            print("✅ Consolidation complete.")

        # --- 5. Save the final dataset ---
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Success! Saved final file with sentences to '{output_path}'")
        return output_path # Return path for the next step

    except Exception as e:
        print(f"An error occurred during the sentence mapping process: {e}")
        return None

def impute_sentences_by_gloss(data_path, output_path):
    """
    Fills in missing sentences for rows with duplicate glosses.

    Args:
        data_path (str): Path to the data file (e.g., final_train_with_sentences.csv).
        output_path (str): Path to save the new file with imputed sentences.
    """
    print(f"--- ✨ Imputing missing sentences for {os.path.basename(data_path)} ---")
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Input file not found: {data_path}")
        return
    
    try:
        df = pd.read_csv(data_path)
        
        # Create a mapping of gloss -> sentence for all non-empty sentences
        gloss_to_sentence_map = df.dropna(subset=['sentence']).set_index('gloss')['sentence'].to_dict()
        
        # Find where sentences are null
        missing_sentence_mask = df['sentence'].isnull()
        
        # Apply the map to fill in the missing sentences
        df.loc[missing_sentence_mask, 'sentence'] = df.loc[missing_sentence_mask, 'gloss'].map(gloss_to_sentence_map)
        
        print(f"✅ Imputation complete. Filled in missing sentences where possible.")
        df.to_csv(output_path, index=False)
        print(f"✅ Success! Saved final imputed file to '{output_path}'")

    except Exception as e:
        print(f"An error occurred during the imputation process: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- 1. Define the input file paths ---
    train_a_path = '/content/drive/MyDrive/medCSLR/subset/subset_train.csv'
    dev_a_path = '/content/drive/MyDrive/medCSLR/subset/subset_dev.csv'
    train_b_path = '/content/drive/MyDrive/medCSLR/data/train_sentence.csv'
    dev_b_path = '/content/drive/MyDrive/medCSLR/data/dev_sentence.csv'

    # --- 2. Define intermediate paths ---
    final_train_path = '/content/drive/MyDrive/medCSLR/final_data/final_train.csv'
    final_dev_path = '/content/drive/MyDrive/medCSLR/final_data/final_dev.csv'
    
    # --- 3. Run the merging and deduplication process ---
    merge_and_deduplicate(train_a_path, train_b_path, final_train_path)
    print("-" * 50)
    merge_and_deduplicate(dev_a_path, dev_b_path, final_dev_path)
    print("\n" + "="*50 + "\n")

    # --- 4. Define paths for sentence mapping ---
    source_sentences_csv = "/content/drive/MyDrive/saudi/saudi-signfor-all-competition/SSL.keypoints.train_signers_train_sentences_prepared.csv"
    train_with_sentences_path = '/content/drive/MyDrive/medCSLR/final_data/final_train_with_sentences.csv'
    dev_with_sentences_path = '/content/drive/MyDrive/medCSLR/final_data/final_dev_with_sentences.csv'

    # --- 5. Run the sentence mapping process ---
    add_sentences_to_final_data(final_train_path, source_sentences_csv, train_with_sentences_path)
    print("-" * 50)
    add_sentences_to_final_data(final_dev_path, source_sentences_csv, dev_with_sentences_path)
    print("\n" + "="*50 + "\n")

    # --- 6. Define final output paths for imputation ---
    final_imputed_train_path = '/content/drive/MyDrive/medCSLR/final_data/final_train_imputed.csv'
    final_imputed_dev_path = '/content/drive/MyDrive/medCSLR/final_data/final_dev_imputed.csv'

    # --- 7. Run the new imputation step ---
    impute_sentences_by_gloss(train_with_sentences_path, final_imputed_train_path)
    print("-" * 50)
    impute_sentences_by_gloss(dev_with_sentences_path, final_imputed_dev_path)
