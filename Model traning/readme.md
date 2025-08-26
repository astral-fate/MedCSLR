 
The `blue-score.py` script is a **text-to-text translation model**. Its specific job is to take a sequence of sign language **glosses** (the written-out names of signs) and translate them into a grammatically correct, natural-sounding Arabic sentence.

Think of it as a two-step pipeline for understanding sign language:

1.  **`cslrconformer.py` (Vision to Gloss):** This model *watches* the video (by analyzing the keypoint data) and identifies the sequence of signs being performed. Its output is a string of glosses, like "أنا إسم محمد". This output is a literal, word-for-word transcription of the signs, which often lacks proper grammar.
2.  **`blue.py` (Gloss to Sentence):** This model takes the output from the first step ("أنا إسم محمد") and translates it into a proper sentence, like "اسمي هو محمد."

---
## What the `blue.py` Code Does

The script fine-tunes a large, pre-trained language model called **mBART** (multilingual BART) for this specific translation task. Here’s a summary of its actions:

1.  **Loads a Pre-trained Model:** It starts by loading `facebook/mbart-large-50-many-to-many-mmt` from Hugging Face. This is a powerful model that already understands the grammar and structure of many languages, including Arabic.
2.  **Prepares Text Data:** It reads `.csv` files that contain pairs of data: a `gloss` column and a `sentence` column. This is its training material.
3.  **Fine-Tuning:** The script then "fine-tunes" the mBART model. This means it continues training the already-smart model on this very specific dataset, teaching it the unique patterns of how to convert sign language glosses into fluent Arabic sentences.
4.  **Evaluation:** It uses the **BLEU (Bilingual Evaluation Understudy)** score to measure performance. The name of the file, `blue.py`, is likely a reference to this metric. BLEU is a standard way to automatically assess the quality of a machine-translated text by comparing it to a high-quality human translation.

---
## Why Keypoints Are Not Used

The reason `blue.py` doesn't use the keypoint data from the `.pkl` or `.pt` files is that **its job begins after the visual analysis is already finished**.

* **`cslrconformer.py` is the Computer Vision (CV) model.** Its input is numerical pose data (the keypoints), and its output is text (the glosses). It answers the question, "What signs do I see?"
* **`blue.py` is the Natural Language Processing (NLP) model.** Its input is text (the glosses), and its output is also text (the final sentence). It answers the question, "How would a person actually say this?"

It operates on a different type of data and serves a different purpose in the overall sign language translation pipeline.

**In short, the two scripts form a sequence:** ➡️

**Video Keypoints** `->` **`cslrconformer.py`** `->` **Glosses** `->` **`blue.py`** `->` **Final Sentence**
