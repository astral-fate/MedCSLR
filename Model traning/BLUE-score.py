# -*- coding: utf-8 -*-
"""
Gloss_to_Sentence_Finetune.ipynb

This script fine-tunes a large, pre-trained multilingual translation model (mBART)
on the specific task of translating sign language glosses into natural Arabic sentences.

Key Steps:
1.  Load a powerful pre-trained model and tokenizer from Hugging Face.
2.  Prepare the medical gloss-sentence pairs for training.
3.  Fine-tune the model on the small, specialized dataset.
4.  Evaluate the model's performance using the BLEU score.
5.  Save the best-performing fine-tuned model.
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# It's recommended to install these dependencies in your Colab environment
# !pip install transformers sentencepiece accelerate
# !pip install evaluate sacrebleu

# BUG FIX: Replaced deprecated torchtext.bleu_score with the official Hugging Face evaluate library
import evaluate
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class CFG:
    # --- PATHS ---
    base_data_path = '/content/drive/MyDrive/ICCV_MSLR_Task1_Data/csv_files/si/'
    train_csv_path = os.path.join(base_data_path, 'train.csv')
    dev_csv_path = os.path.join(base_data_path, 'dev.csv') # Original dev path

    # New paths for the specific medical subset CSVs containing IDs
    medical_train_ids_path = '/content/drive/MyDrive/medCSLR/data/ishara_med/final_train.csv'
    medical_dev_ids_path = '/content/drive/MyDrive/medCSLR/data/ishara_med/final_dev.csv' # Assuming 'final_test' was a typo for 'final_dev'


    work_dir = "/content/drive/MyDrive/CSLR_Experiments/Medical_Translation_Finetune"

    # --- Model ---
    # Using a powerful, pre-trained multilingual model from Hugging Face
    pretrained_model_name = "facebook/mbart-large-50-many-to-many-mmt"

    # --- Fine-Tuning Hyperparameters ---
    finetune_epochs = 50
    batch_size = 4  # Small batch size is crucial for large models
    finetune_lr = 5e-6 # Very low learning rate for fine-tuning
    patience = 10
    max_length = 128 # Max token length for input and output

    # --- System ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

cfg = CFG()

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        # The source language for glosses can be treated as English for tokenization purposes
        self.tokenizer.src_lang = "en_XX"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gloss = str(row.gloss)
        sentence = str(row.sentence)

        # Tokenize the gloss (source)
        model_inputs = self.tokenizer(
            gloss,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize the sentence (target)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                sentence,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids

        # mBART requires labels to be -100 where padding is, so it ignores them in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        # Squeeze to remove the batch dimension that tokenizer adds
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        return model_inputs

# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================
def train_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Fine-Tuning Epoch"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device, metric):
    model.eval()
    all_preds, all_gt = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Generate predictions
        generated_ids = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=cfg.max_length,
            num_beams=5, # Beam search for better quality
            forced_bos_token_id=tokenizer.lang_code_to_id["ar_AR"]
        )

        # Decode predictions
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Prepare ground truth labels for decoding
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        gt = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metric.add_batch(predictions=preds, references=gt)

    return metric.compute()


def main_finetune_translation():
    print("--- 🩺 Starting Gloss-to-Sentence Translation Fine-Tuning ---")
    os.makedirs(cfg.work_dir, exist_ok=True)

    # --- 1. Load Model and Tokenizer ---
    print(f"Loading pre-trained model: {cfg.pretrained_model_name}")
    tokenizer = MBart50TokenizerFast.from_pretrained(cfg.pretrained_model_name)
    model = MBartForConditionalGeneration.from_pretrained(cfg.pretrained_model_name).to(cfg.device)

    # --- 2. Load and Prepare Data ---
    try:
        # Load the full datasets
        train_df_full = pd.read_csv(cfg.train_csv_path)
        dev_df_full = pd.read_csv(cfg.dev_csv_path)

        # Load the medical subset IDs from the specified files
        if not os.path.exists(cfg.medical_train_ids_path) or not os.path.exists(cfg.medical_dev_ids_path):
             print(f"❌ ERROR: Medical subset ID files not found.")
             if not os.path.exists(cfg.medical_train_ids_path): print(f"  - Missing: {cfg.medical_train_ids_path}")
             if not os.path.exists(cfg.medical_dev_ids_path): print(f"  - Missing: {cfg.medical_dev_ids_path}")
             return

        medical_train_df = pd.read_csv(cfg.medical_train_ids_path)
        medical_dev_df = pd.read_csv(cfg.medical_dev_ids_path)

        if 'id' not in medical_train_df.columns or 'id' not in medical_dev_df.columns:
             print("\n❌ ERROR: 'id' column not found in your medical subset CSV files.")
             return

        # Filter the full datasets using the IDs from the medical subset files
        medical_train_ids = medical_train_df['id'].astype(str).tolist()
        medical_dev_ids = medical_dev_df['id'].astype(str).tolist()

        medical_train_df_filtered = train_df_full[train_df_full['id'].astype(str).isin(medical_train_ids)].copy()
        medical_dev_df_filtered = dev_df_full[dev_df_full['id'].astype(str).isin(medical_dev_ids)].copy()


        if 'sentence' not in medical_train_df_filtered.columns or 'sentence' not in medical_dev_df_filtered.columns:
            print("\n❌ ERROR: 'sentence' column not found in the filtered medical datasets.")
            print("Please ensure the original train.csv and dev.csv files contain the 'sentence' column.")
            return

    except FileNotFoundError:
        print(f"❌ ERROR: Original CSV files not found at {cfg.base_data_path}")
        return

    train_dataset = TranslationDataset(medical_train_df_filtered, tokenizer, cfg.max_length)
    val_dataset = TranslationDataset(medical_dev_df_filtered, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    print(f"Medical Train subset size: {len(train_dataset)}")
    print(f"Medical Validation subset size: {len(val_dataset)}")


    # --- 3. Fine-Tuning Loop ---
    print("\n--- Starting Fine-Tuning ---")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.finetune_lr)
    scaler = torch.cuda.amp.GradScaler()

    # Load the BLEU metric from the evaluate library
    bleu_metric = evaluate.load('sacrebleu')

    best_bleu = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(cfg.work_dir, "best_medical_translation_model.pt")

    for epoch in range(cfg.finetune_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, cfg.device, scaler)
        results = evaluate(model, val_loader, tokenizer, cfg.device, bleu_metric)
        bleu_score = results['score']

        print(f"Epoch {epoch+1}/{cfg.finetune_epochs} | Train Loss: {train_loss:.4f} | Val BLEU: {bleu_score:.2f}")

        if bleu_score > best_bleu:
            best_bleu = bleu_score
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best model saved with BLEU score: {best_bleu:.2f}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(f"Stopping early. No improvement in {cfg.patience} epochs.")
            break

    print(f"\n--- ✅ Fine-Tuning Complete ---")
    print(f"Best validation BLEU score: {best_bleu:.2f}")
    print(f"Best model saved to: {checkpoint_path}")

if __name__ == '__main__':
    main_finetune_translation()
