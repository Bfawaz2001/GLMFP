import gc
import pickle
import torch
from torch.nn.utils import clip_grad_norm_
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
    label_encoder.fit(list("ABCDEFGHIKLMNOPQRSTUVWXYZ"))
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


def train(model, train_loader, val_loader, optimizer, criterion, epochs, model_path, label_encoder):
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    best_val_loss = float('inf')
    scaler = GradScaler()

    # Define the number of steps for gradient accumulation
    accumulation_steps = 4
    accumulation_counter = 0  # Counter to keep track of accumulation steps

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Reset optimizer gradients at the start of each epoch
        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Autocast context for mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets) / accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()
            train_loss += loss.item() * accumulation_steps  # Scale loss back

            accumulation_counter += 1

            # Perform model update steps if accumulation counter has reached the defined steps
            if accumulation_counter % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                # Adjust gradients if necessary (gradient clipping, etc.)
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()

                # Zero gradients
                optimizer.zero_grad()

                # Reset accumulation counter after model update
                accumulation_counter = 0

            # Memory cleanup after each batch
            del inputs, targets, outputs, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.transpose(1, 2), targets)

                val_loss += loss.item()

                # Memory cleanup in validation phase
                del inputs, targets, outputs, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_loss_avg = val_loss / len(val_loader)

        # Step scheduler
        scheduler.step(val_loss_avg)

        # Logging
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader)}, '
              f'Validation Loss: {val_loss_avg}')

        # Model checkpointing
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), model_path)
            with open("nn_label_encoder.pkl", 'wb') as f:
                pickle.dump(label_encoder, f)
            print("Model and LabelEncoder saved.")

    print(f"Training completed. Best model saved to {model_path}.")


def main(fasta_file, model_path):
    """Main function to run the training process."""
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    encoded_seqs, label_encoder_classes = encode_sequences(sequences)
    label_encoder = LabelEncoder()
    vocab_size = len(set("ACDEFGHIKLMNPQRSTVWYXZBUO")) + 1

    train_seqs, val_seqs = train_test_split(encoded_seqs, test_size=0.25, random_state=101)
    print("Building model...")
    train_dataset = ProteinSequenceDataset(train_seqs)
    val_dataset = ProteinSequenceDataset(val_seqs)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate,
                              pin_memory=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate, pin_memory=True,
                            num_workers=1)

    model = LSTMProteinGenerator(vocab_size, embedding_dim=64, hidden_dim=128, num_layers=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, epochs=10, model_path=model_path,
          label_encoder=label_encoder)


if __name__ == "__main__":
    fasta_file = "uniprot_sprot.fasta"  # Update the path as necessary
    model_path = "nn_pytorch.pt"  # Update the path as necessary
    main(fasta_file, model_path)