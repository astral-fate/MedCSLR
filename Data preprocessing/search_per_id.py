import pandas as pd
import os
import json

def search_for_id_in_file(dataset_path, target_id):
    """
    Searches for a specific ID in a CSV or JSON file and prints its corresponding sentence.

    Args:
        dataset_path (str): Path to the file to search within.
        target_id (str): The ID to search for.
    """
    print(f"--- 🔍 Searching in: {os.path.basename(dataset_path)} ---")

    # --- 1. Check if the file exists ---
    if not os.path.exists(dataset_path):
        print(f"❌ File not found.")
        return

    # --- 2. Skip non-text files like H5 ---
    file_extension = os.path.splitext(dataset_path)[1]
    if file_extension not in ['.csv', '.json']:
        print(f"⚪ Skipping binary file format ({file_extension}).")
        return

    try:
        # --- 3. Load the dataset based on its type ---
        df = None
        if file_extension == '.csv':
            df = pd.read_csv(dataset_path)
        elif file_extension == '.json':
            # JSON files can have different structures, this handles common ones.
            # It might need adjustment if the JSON is nested differently.
            df = pd.read_json(dataset_path)

        # --- 4. Standardize column names for searching ---
        # Check for common variations of 'id' and 'sentence' columns
        rename_map = {}
        if 'ID' in df.columns: rename_map['ID'] = 'id'
        if 'Translation' in df.columns: rename_map['Translation'] = 'sentence'
        df.rename(columns=rename_map, inplace=True)

        if 'id' not in df.columns:
            print(f"❌ ERROR: Could not find a recognizable ID column in this file.")
            return

        # --- 5. Search for the ID ---
        result_row = df[df['id'] == target_id]

        # --- 6. Report the result ---
        if not result_row.empty:
            print(f"✅ Success! ID '{target_id}' was found.")
            # Check if a sentence column exists to print
            if 'sentence' in result_row.columns:
                sentence = result_row['sentence'].iloc[0]
                print(f"   - Corresponding Sentence: '{sentence}'")
        else:
            print(f"❌ Not Found: The ID '{target_id}' does not exist in this file.")

    except Exception as e:
        print(f"An error occurred while processing this file: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Base directory where the files are located
    base_path = "/content/drive/MyDrive/saudi/saudi-signfor-all-competition/"

    # List of all files to search through
    files_to_search = [
        "SSL.keypoints.test_signers_test_sentences.json",
        "SSL.keypoints.train_signers_train_sentences_prepared.csv",
        "SSL.keypoints.train_signers_train_sentences.csv",
        "SSL.keypoints.test_signers_test_sentences.h5",
        "SSL.keypoints.test_signers_test_sentences.hands_only.h5",
        "SSL.keypoints.train_signers_train_sentences.0.h5",
        "SSL.keypoints.train_signers_train_sentences.hands_only.h5"
    ]

    # The specific ID you want to find
    id_to_find = "00_0953"

    print(f"--- Starting search for ID: '{id_to_find}' ---")
    # --- Loop through the files and run the search function ---
    for file_name in files_to_search:
        full_path = os.path.join(base_path, file_name)
        search_for_id_in_file(full_path, id_to_find)
        print("-" * 40) # Add a separator for clarity
