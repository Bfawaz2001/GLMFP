import gc
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

# Setup device: Uses Metal if available (for Mac), otherwise defaults to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)


class ProteinSequenceDataset(Dataset):
    """Custom Dataset for loading and transforming protein sequences for sequence generation."""

    def __init__(self, sequences, seq_len=50):
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
    # Include all standard amino acids, plus some special characters for padding, etc.
    label_encoder.fit(list("ACDEFGHIKLMNPQRSTVWYXZBJUO"))
    encoded_seqs = [torch.tensor(label_encoder.transform(list(seq))) for seq in sequences]
    return encoded_seqs, label_encoder.classes_


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


def train(model, train_loader, val_loader, optimizer, criterion, epochs, model_path):
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()

            # Clip gradients to avoid exploding gradient issue
            clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            train_loss += loss.item()

            # Memory management
            del inputs, targets, outputs, loss
            gc.collect()
            if device == torch.device("mps"):
                torch.mps.empty_cache()
            if device == torch.device("gpu"):
                torch.gpu.empty_cache()
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()



        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets)
                val_loss += loss.item()

                # Memory management
                del inputs, targets, outputs, loss
                gc.collect()
                if device == torch.device("mps"):
                    torch.mps.empty_cache()

        val_loss_avg = val_loss / len(val_loader)
        scheduler.step(val_loss_avg)

        # Print training/validation statistics
        print(
            f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss_avg}')

        # Save the model if validation loss has improved
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), model_path)

    print(f"Training completed. Best model saved to {model_path}.")

def main(fasta_file, model_path):
    """Main function to run the training process."""
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    encoded_seqs, _ = encode_sequences(sequences)
    vocab_size = len(set("ACDEFGHIKLMNPQRSTVWYXZBJUO")) + 1

    train_seqs, val_seqs = train_test_split(encoded_seqs, test_size=0.2, random_state=42)
    print("Building model...")
    train_dataset = ProteinSequenceDataset(train_seqs)
    val_dataset = ProteinSequenceDataset(val_seqs)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate,
                              pin_memory=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate, pin_memory=True,
                            num_workers=4)

    model = LSTMProteinGenerator(vocab_size, embedding_dim=32, hidden_dim=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, epochs=10, model_path=model_path)


if __name__ == "__main__":
    fasta_file = "../../../data/training data/uniprot_sprot.fasta"  # Update the path as necessary
    model_path = "../../../models/lstm_protein_generator.pt"  # Update the path as necessary
    main(fasta_file, model_path)
