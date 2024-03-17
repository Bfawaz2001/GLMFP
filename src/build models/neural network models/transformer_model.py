import gc
import math
import pickle
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

# Setup device: use CUDA for GPU acceleration if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'}")


class ProteinSequenceDataset(Dataset):
    """Custom Dataset for loading and transforming protein sequences."""
    def __init__(self, sequences, seq_len=50):
        self.sequences = sequences
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = seq[:-1]  # Input sequence
        target_seq = seq[1:]  # Target sequence
        return input_seq, target_seq


def encode_sequences(sequences):
    """Encode protein sequences using Label Encoding."""
    label_encoder = LabelEncoder()
    label_encoder.fit(list("ACDEFGHIKLMNPQRSTVWYXZBUO"))  # Extended amino acid alphabet
    encoded_seqs = [torch.tensor(label_encoder.transform(list(seq))) for seq in sequences]
    return encoded_seqs, label_encoder.classes_


def pad_collate(batch):
    """Collate function to pad sequences for batch processing."""
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    return input_seqs_padded, target_seqs_padded


class TransformerProteinGenerator(nn.Module):
    """Transformer model for protein sequence generation with dynamic positional encoding."""

    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout=0.1)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        seq_len, N = src.size(1), src.size(0)

        # Generate dynamic positional encoding
        pos_encoder = self.positional_encoding(seq_len, N, self.d_model).to(src.device)

        src = self.embedding(src) * math.sqrt(self.d_model) + pos_encoder
        tgt = self.embedding(tgt) * math.sqrt(self.d_model) + pos_encoder

        output = self.transformer(src, tgt)
        return self.fc_out(output)

    def positional_encoding(self, seq_len, batch_size, d_model):
        """Generates dynamic positional encodings."""
        pe = torch.zeros(seq_len, batch_size, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe.permute(1, 0, 2)


def train(model, train_loader, val_loader, optimizer, criterion, epochs, model_path, label_encoder):
    """Training loop for the Transformer model."""
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    best_val_loss = float('inf')
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs, inputs)
                loss = criterion(outputs.transpose(1, 2), targets)

            scaler.scale(loss).backward()
            train_loss += loss.item()

            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            del inputs, targets, outputs, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                with autocast():
                    outputs = model(inputs, inputs)
                    loss = criterion(outputs.transpose(1, 2), targets)

                val_loss += loss.item()

                del inputs, targets, outputs, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_loss_avg = val_loss / len(val_loader)
        scheduler.step(val_loss_avg)

        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader)}, '
              f'Validation Loss: {val_loss_avg}')

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), model_path)
            with open("transformer_label_encoder.pkl", 'wb') as f:
                pickle.dump(label_encoder, f)
            print("Model and LabelEncoder saved.")

    print(f"Training completed. Best model saved to {model_path}.")


def main(fasta_file, model_path):
    """Main function to execute the model training."""
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    encoded_seqs, label_encoder_classes = encode_sequences(sequences)
    label_encoder = LabelEncoder()
    vocab_size = len(label_encoder_classes) + 1  # Plus one for padding

    train_seqs, val_seqs = train_test_split(encoded_seqs, test_size=0.25, random_state=101)

    train_dataset = ProteinSequenceDataset(train_seqs)
    val_dataset = ProteinSequenceDataset(val_seqs)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate, pin_memory=True,
                              num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate, pin_memory=True,
                            num_workers=2)

    model = TransformerProteinGenerator(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, epochs=50, model_path=model_path,
          label_encoder=label_encoder)


if __name__ == "__main__":
    fasta_file = "uniprot_sprot.fasta"  # Update path as necessary
    model_path = "transformer_model.pth"  # Update path as necessary
    main(fasta_file, model_path)
