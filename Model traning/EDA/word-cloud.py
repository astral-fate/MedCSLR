# First, you need to install the required libraries.
# You can do this by running the following commands in your terminal or a code cell:
# pip install pandas
# pip install wordcloud
# pip install arabic_reshaper
# pip install python-bidi

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

def generate_arabic_wordcloud(csv_paths, column_name='gloss', font_path=None):
    """
    Reads one or more CSV files, combines the text from a specified column,
    and generates a word cloud for Arabic text.

    Args:
        csv_paths (list): A list of file paths to the CSV files.
        column_name (str): The name of the column containing the text data.
        font_path (str, optional): The path to a .ttf font file that supports Arabic.
                                   If None, it tries a default path.
    """
    if not csv_paths:
        print("Error: Please provide a list of CSV file paths.")
        return

    try:
        # --- 1. Load and Combine Data ---
        # Create a list to hold the DataFrames
        df_list = []
        for path in csv_paths:
            df_list.append(pd.read_csv(path))

        # Concatenate all DataFrames into a single one
        combined_df = pd.concat(df_list, ignore_index=True)

        # --- 2. Prepare the Text ---
        # Combine all the text from the specified column into a single string
        # We use ' '.join() to separate the words from different rows with a space
        text = ' '.join(combined_df[column_name].dropna().astype(str))

        # --- 3. Reshape Arabic Text for Display ---
        # This is a crucial step for rendering Arabic characters correctly
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)

        # --- 4. Generate the Word Cloud ---
        # You must specify a font that supports Arabic characters.
        # 'Arial' is a common one. If it's not found, you may need to provide
        # a direct path to a .ttf file on your system.
        # For Google Colab, a common path is '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        # but a dedicated Arabic font like 'Amiri-Regular.ttf' is better if installed.
        if font_path is None:
            # Attempt to use a commonly available font.
            # Replace 'Arial' with the name or path of an Arabic font on your system if needed.
            font_path = 'Arial'

        print("Generating word cloud...")
        wordcloud = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color='white',
            collocations=False, # Avoids grouping words into bigrams
        ).generate(bidi_text)
        print("Word cloud generated successfully.")


        # --- 5. Display the Word Cloud ---
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off") # Hide the axes
        plt.title("Most Frequent Words in Medical Glosses", fontsize=20)
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Please check your file paths.")
    except KeyError:
        print(f"Error: Column '{column_name}' not found in one of the CSV files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main Execution ---
if __name__ == '__main__':
    # Define the paths to your CSV files
    # IMPORTANT: Replace these with the actual paths to your files
    medical_train_ids_path = '/content/drive/MyDrive/medCSLR/data/ishara_med/train.csv'
    medical_dev_ids_path = '/content/drive/MyDrive/medCSLR/data/ishara_med/dev.csv'

    # Create a list of the file paths
    all_csv_paths = [medical_train_ids_path, medical_dev_ids_path]

    # Call the function to generate and display the word cloud
    # If you have a specific Arabic font file (.ttf), you can specify its path like this:
    # generate_arabic_wordcloud(all_csv_paths, font_path='/path/to/your/arabic_font.ttf')
    generate_arabic_wordcloud(all_csv_paths)
