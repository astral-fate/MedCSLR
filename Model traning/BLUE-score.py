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

import evaluate
from evaluate import load
import sacrebleu

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class CFG:
    # --- PATHS ---
    base_data_path = '/content/drive/MyDrive/medCSLR/data/ishara/'
    train_csv_path = os.path.join(base_data_path, 'train.csv')
    dev_csv_path = os.path.join(base_data_path, 'dev.csv')

    medical_train_ids_path = '/content/drive/MyDrive/medCSLR/data/ishara_med/final_train.csv'
    medical_dev_ids_path = '/content/drive/MyDrive/medCSLR/data/ishara_med/final_dev.csv'

    work_dir = "/content/drive/MyDrive/CSLR_Experiments/Medical_Translation_Finetune"

    # --- Model ---
    pretrained_model_name = "facebook/mbart-large-50-many-to-many-mmt"

    # --- Fine-Tuning Hyperparameters ---
    finetune_epochs = 50
    batch_size = 4
    finetune_lr = 5e-6
    patience = 10
    max_length = 128

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
        # No need to set src_lang here, will be set in __getitem__

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gloss = str(row.gloss)
        sentence = str(row.sentence)

        # ✅ CORRECTED TOKENIZATION
        # Use a single, modern tokenizer call with the `text_target` argument.
        # This is the recommended approach and avoids the deprecated context manager.
        # The tokenizer needs to know the target language for the labels, which is handled
        # by setting the tokenizer's language code before the call.
        self.tokenizer.src_lang = "ar_AR" # Source is Arabic gloss
        self.tokenizer.tgt_lang = "ar_AR" # Target is Arabic sentence

        model_inputs = self.tokenizer(
            text=gloss,
            text_target=sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # The tokenizer output now includes 'labels'. We just need to process them.
        model_inputs['labels'][model_inputs['labels'] == self.tokenizer.pad_token_id] = -100
        
        # Squeeze the tensors to remove the batch dimension (which is 1)
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

        with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
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
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        generated_ids = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=cfg.max_length,
            num_beams=5,
            # Force the model to start generating in Arabic
            forced_bos_token_id=tokenizer.lang_code_to_id["ar_AR"]
        )
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        gt = tokenizer.batch_decode(labels, skip_special_tokens=True)
        wrapped_gt = [[g] for g in gt]
        metric.add_batch(predictions=preds, references=wrapped_gt)

    return metric.compute()


def main_finetune_translation():
    print("--- 🩺 Starting Gloss-to-Sentence Translation Fine-Tuning ---")
    os.makedirs(cfg.work_dir, exist_ok=True)

    print(f"Loading pre-trained model: {cfg.pretrained_model_name}")
    tokenizer = MBart50TokenizerFast.from_pretrained(cfg.pretrained_model_name)
    model = MBartForConditionalGeneration.from_pretrained(cfg.pretrained_model_name).to(cfg.device)

    try:
        train_df_full = pd.read_csv(cfg.train_csv_path)
        dev_df_full = pd.read_csv(cfg.dev_csv_path)

        medical_train_ids_path = '/content/drive/MyDrive/medCSLR/data/ishara_med/train.csv'
        medical_dev_ids_path = '/content/drive/MyDrive/medCSLR/data/ishara_med/dev.csv'
        
        medical_train_df = pd.read_csv(medical_train_ids_path)
        medical_dev_df = pd.read_csv(medical_dev_ids_path)

        medical_train_ids = medical_train_df['id'].astype(str).tolist()
        medical_dev_ids = medical_dev_df['id'].astype(str).tolist()

        medical_train_df_filtered = train_df_full[train_df_full['id'].astype(str).isin(medical_train_ids)].copy()
        medical_dev_df_filtered = dev_df_full[dev_df_full['id'].astype(str).isin(medical_dev_ids)].copy()

        # --- Data Cleaning Section (Good Practice) ---
        print(f"\n--- Data Cleaning ---")
        for name, df in [("Train", medical_train_df_filtered), ("Dev", medical_dev_df_filtered)]:
            print(f"Original {name} size: {len(df)}")
            df['gloss'] = df['gloss'].astype(str)
            df['sentence'] = df['sentence'].astype(str)
            df.dropna(subset=['gloss', 'sentence'], inplace=True)
            df.drop(df[df['gloss'].str.strip() == ""].index, inplace=True)
            df.drop(df[df['sentence'].str.strip() == ""].index, inplace=True)
            print(f"Cleaned {name} size: {len(df)}")
        # ----------------------------------------------

    except FileNotFoundError:
        print(f"❌ ERROR: Original CSV files not found.")
        return

    train_dataset = TranslationDataset(medical_train_df_filtered, tokenizer, cfg.max_length)
    val_dataset = TranslationDataset(medical_dev_df_filtered, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    print(f"\nMedical Train subset size: {len(train_dataset)}")
    print(f"Medical Validation subset size: {len(val_dataset)}")

    print("\n--- Starting Fine-Tuning ---")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.finetune_lr)
    scaler = torch.amp.GradScaler(enabled=(cfg.device == 'cuda'))
    bleu_metric = load('sacrebleu')
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
