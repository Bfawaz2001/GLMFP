import pickle
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

# Setup device: use Cuda for GPU acceleration otherwise use the CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")


class ProteinSequenceDataset(Dataset):
    """Custom Dataset for loading and transforming protein sequences for sequence generation."""
    def __init__(self, sequences, seq_len=2500):
        self.sequences = sequences
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = seq[:-1]  # Input sequence for the model
        target_seq = seq[1:]  # Target sequence for the model
        return input_seq, target_seq


def encode_sequences(sequences):
    """Encode protein sequences into numerical format using Label Encoding."""
    label_encoder = LabelEncoder()
    label_encoder.fit(list("ABCDEFGHIKLMNOPQRSTUVWXYZ"))
    encoded_seqs = [torch.tensor(label_encoder.transform(list(seq))) for seq in sequences]
    return encoded_seqs, label_encoder


def pad_collate(batch):
    """Collate function to pad sequences for batch processing."""
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    return input_seqs_padded, target_seqs_padded


class LSTMProteinGenerator(nn.Module):
    """LSTM model for protein sequence generation."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMProteinGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out)
        return out


def train(model, train_loader, val_loader, optimizer, criterion, epochs, model_path, label_encoder):
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    best_val_loss = float('inf')
    scaler = GradScaler()  # For AMP

    with open("../../../data/models/neural network/nn_label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder saved as nn_label_encoder.pkl.")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Clear gradients

            # Using AMP for reduced memory usage
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets)

            scaler.scale(loss).backward()  # Scale the loss before backward
            scaler.step(optimizer)  # Scale for optimizer
            scaler.update()  # Update scale for next iteration

            train_loss += loss.item()

        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad(), autocast():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)
        scheduler.step(val_loss_avg)  # Adjust learning rate

        # Logging
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader)}, '
              f'Validation Loss: {val_loss_avg}')

        # Model checkpointing
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), model_path)
            print("Model saved.")

    print(f"Training completed. Best model saved to {model_path}.")


def main(fasta_file, model_path):
    """Main function to run the training process."""
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    encoded_seqs, label_encoder = encode_sequences(sequences)
    vocab_size = len(label_encoder.classes_) + 1

    train_seqs, val_seqs = train_test_split(encoded_seqs, test_size=0.3, random_state=64)

    print("Building model...")
    train_dataset = ProteinSequenceDataset(train_seqs)
    val_dataset = ProteinSequenceDataset(val_seqs)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate,
                              pin_memory=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, collate_fn=pad_collate,
                            pin_memory=True, num_workers=1)

    model = LSTMProteinGenerator(vocab_size, embedding_dim=64, hidden_dim=128, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training model...")
    train(model, train_loader, val_loader, optimizer, criterion, epochs=50, model_path=model_path,
          label_encoder=label_encoder)


if __name__ == "__main__":
    fasta_file = "../../../data/training data/uniprot_sprot.fasta"  # Update the path as necessary
    model_path = "../../../data/models/neural network/nn_model.pt"  # Update the path as necessary
    main(fasta_file, model_path)