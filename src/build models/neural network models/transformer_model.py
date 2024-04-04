import math
import pickle
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

# Setup device: use Cuda for GPU acceleration otherwise use the CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")


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
    label_encoder.fit(list("ABCDEFGHIKLMNOPQRSTUVWXYZ"))  # Extended amino acid alphabet
    encoded_seqs = [torch.tensor(label_encoder.transform(list(seq))) for seq in sequences]
    return encoded_seqs, label_encoder.classes_


def pad_collate(batch):
    """Collate function to pad sequences for batch processing."""
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    return input_seqs_padded, target_seqs_padded


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape [sequence length, 1, embedding size] for correct broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x expected to be of shape [sequence length, batch size, embedding size]
        x = x + self.pe[:x.size(0), :]  # Corrected indexing for dynamic sequence lengths
        return self.dropout(x)


class TransformerProteinGenerator(nn.Module):
    """Transformer model for protein sequence generation."""
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=6, dropout=0.1):
        super(TransformerProteinGenerator, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


def train(model, train_loader, val_loader, optimizer, criterion, epochs, model_path, label_encoder):
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    best_val_loss = float('inf')
    max_grad_norm = 1.0  # Define maximum gradient norm for clipping
    scaler = GradScaler()  # Initialize the gradient scaler for mixed precision training

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets)

            # Scales the loss, and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            train_loss += loss.item()

            # Gradient clipping
            scaler.unscale_(optimizer)  # Unscales the gradients of optimizer's assigned params in-place
            clip_grad_norm_(model.parameters(), max_grad_norm)

            # optimizer's step is skipped in case the gradients contain infs or NaNs
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            # Optional: Clear memory to prevent OOM
            del inputs, targets, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Validation loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)
        scheduler.step(val_loss_avg)

        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss_avg}')

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

    train_seqs, val_seqs = train_test_split(encoded_seqs, test_size=0.3, random_state=10)

    train_dataset = ProteinSequenceDataset(train_seqs)
    val_dataset = ProteinSequenceDataset(val_seqs)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate, pin_memory=True,
                              num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate, pin_memory=True,
                            num_workers=1)

    model = TransformerProteinGenerator(vocab_size).to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimiser, criterion, epochs=25, model_path=model_path,
          label_encoder=label_encoder)


if __name__ == "__main__":
    main("uniprot_sprot.fasta", "transformer_model.pth")