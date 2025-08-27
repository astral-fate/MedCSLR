# Simplified approach - sometimes less processing is better
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# Load the datasets
try:
    train_df = pd.read_csv('/content/drive/MyDrive/medCSLR/data/ishara_med/train.csv')
    dev_df = pd.read_csv('/content/drive/MyDrive/medCSLR/data/ishara_med/dev.csv')
    combined_df = pd.concat([train_df, dev_df])
    
    # Get all words - minimal processing
    all_text = ' '.join(combined_df['gloss'].dropna())
    words = all_text.split()  # Simple split instead of regex
    
    # Clean words (remove punctuation but keep Arabic)
    cleaned_words = []
    for word in words:
        # Remove punctuation but keep Arabic and English characters
        clean_word = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\w]', '', word)
        if clean_word:  # Only add non-empty words
            cleaned_words.append(clean_word)
    
    word_counts = Counter(cleaned_words)
    
    # Try creating wordcloud without any Arabic reshaping
    # Sometimes the original text is correct and reshaping breaks it
    wordcloud_simple = WordCloud(
        font_path='/content/Amiri-Regular.ttf',
        background_color='white',
        width=1200,
        height=600,
        max_words=100,
        collocations=False,
        # Try RTL support
        prefer_horizontal=0.9
    )
    
    # Method 1: Direct from text
    simple_text = ' '.join([word * count for word, count in word_counts.items()])
    wc1 = wordcloud_simple.generate(simple_text)
    
    # Method 2: From frequencies
    wc2 = wordcloud_simple.generate_from_frequencies(word_counts)
    
    # Display both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.imshow(wc1, interpolation='bilinear')
    ax1.set_title('Method 1: From Text')
    ax1.axis('off')
    
    ax2.imshow(wc2, interpolation='bilinear')
    ax2.set_title('Method 2: From Frequencies')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_arabic_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Top 15 words (no processing):")
    for word, count in word_counts.most_common(15):
        print(f"'{word}': {count}")
    
    # Check if words contain Arabic characters
    arabic_words = [word for word in cleaned_words if re.search(r'[\u0600-\u06FF]', word)]
    print(f"\nTotal words: {len(cleaned_words)}")
    print(f"Words containing Arabic characters: {len(arabic_words)}")
    
    if arabic_words:
        print(f"Sample Arabic words: {arabic_words[:10]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
