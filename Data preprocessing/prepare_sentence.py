import pandas as pd
import os

def analyze_and_prepare_dataset(file_path, output_path):
    """
    Performs EDA on a CSV file, checks for a sentence column,
    and prepares it for the translation task by renaming the column.

    Args:
        file_path (str): The exact path to the source CSV file.
        output_path (str): The path to save the modified CSV file.
    """
    print(f"--- 🕵️‍♂️ Starting Analysis for: {file_path} ---")

    # --- 1. Check if the file exists ---
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found at the specified path.")
        print("Please ensure the file path is correct and the file is accessible.")
        return

    try:
        # --- 2. Load the dataset ---
        df = pd.read_csv(file_path)
        print("✅ File loaded successfully.")

        # --- 3. Display Logs ---
        print("\n--- 📜 LOGS ---")
        print("\n[LOG] DataFrame Information:")
        from io import StringIO
        buffer = StringIO()
        df.info(buf=buffer)
        print(buffer.getvalue())

        print("\n[LOG] First 5 Rows of the DataFrame:")
        print(df.head().to_string())

        # --- 4. Analysis and Conclusion ---
        print("\n--- 📊 CONCLUSION & ACTION ---")
        
        # Based on your logs, the sentence column is 'Translation'
        if 'Translation' in df.columns:
            print("✅ Success: The 'Translation' column was found.")
            
            # Rename the column to 'sentence' for compatibility
            print("\nRenaming 'Translation' column to 'sentence'...")
            df.rename(columns={'Translation': 'sentence', 'ID': 'id'}, inplace=True)
            
            # Save the modified DataFrame
            print(f"Saving the updated data to: {output_path}")
            df.to_csv(output_path, index=False)
            
            print("\n✅ New file saved successfully with the correct column names ('id', 'sentence').")
            print("You can now use this new file for your training script.")

        elif 'sentence' in df.columns:
            print("✅ Success: The 'sentence' column already exists. No action needed.")
        
        else:
            print("❌ Not Found: Neither 'Translation' nor 'sentence' column was found.")
            print(f"\nAvailable columns are: {df.columns.to_list()}")
            print("Please check the CSV file to identify the correct column for sentences.")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        print("The file might be corrupted or not in a standard CSV format.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Path to the original training CSV file you mentioned.
    input_csv_path = "/content/drive/MyDrive/saudi/saudi-signfor-all-competition/SSL.keypoints.train_signers_train_sentences.csv"
    
    # Path for the new, corrected file.
    output_csv_path = "/content/drive/MyDrive/saudi/saudi-signfor-all-competition/SSL.keypoints.train_signers_train_sentences_prepared.csv"
    
    # Run the analysis and preparation function
    analyze_and_prepare_dataset(input_csv_path, output_csv_path)
