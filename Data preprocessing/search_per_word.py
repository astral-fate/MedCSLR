import pandas as pd
import os
import json

def search_for_keyword_in_file(dataset_path, target_keyword):
    """
    Searches for a specific keyword in the 'sentence' column of a CSV or JSON file.

    Args:
        dataset_path (str): Path to the file to search within.
        target_keyword (str): The keyword to search for in the sentences.
    """
    print(f"--- 🔍 Searching for '{target_keyword}' in: {os.path.basename(dataset_path)} ---")

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
            df = pd.read_json(dataset_path)

        # --- 4. Standardize column names for searching ---
        rename_map = {}
        if 'ID' in df.columns: rename_map['ID'] = 'id'
        if 'Translation' in df.columns: rename_map['Translation'] = 'sentence'
        df.rename(columns=rename_map, inplace=True)

        if 'sentence' not in df.columns:
            print(f"❌ ERROR: Could not find a recognizable sentence column in this file.")
            return

        # --- 5. Search for the keyword in the 'sentence' column ---
        # The search is case-insensitive and handles non-string data gracefully.
        results = df[df['sentence'].str.contains(target_keyword, case=False, na=False)]

        # --- 6. Report the result ---
        if not results.empty:
            print(f"✅ Success! Found {len(results)} instance(s) of '{target_keyword}':")
            # Print each matching row (ID and sentence)
            for index, row in results.iterrows():
                print(f"   - ID: {row.get('id', 'N/A')}, Sentence: '{row['sentence']}'")
        else:
            print(f"❌ Not Found: The keyword '{target_keyword}' was not found in this file.")

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

    # The specific keyword you want to find. "poison" will match "poison" and "poisoning".
    keyword_to_find = "pain"

    print(f"--- Starting search for keyword: '{keyword_to_find}' ---")
    # --- Loop through the files and run the search function ---
    for file_name in files_to_search:
        full_path = os.path.join(base_path, file_name)
        search_for_keyword_in_file(full_path, keyword_to_find)
        print("-" * 40) # Add a separator for clarity
