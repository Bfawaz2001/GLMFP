import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import time

class ProteinSequenceDataset(Dataset):
    """Custom Dataset for loading and transforming protein sequences for sequence generation."""

    def __init__(self, sequences, seq_len=50):
        """
        Initializes the dataset with protein sequences.
        :param sequences: A list of encoded protein sequences.
        :param seq_len: The length of the sequence chunk to use for training.
        """
        self.sequences = sequences
        self.seq_len = seq_len

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index by preparing it as an input-target pair.
        Here, each sequence is split into two parts: the input sequence and the target sequence.
        """
        # Here's a simplistic way to define inputs and targets. You might need to adjust it based on your actual use case.
        seq = self.sequences[idx]
        input_seq = seq[:-1]  # All but the last character
        target_seq = seq[1:]  # All but the first character
        return input_seq, target_seq

def encode_sequences(sequences):
    """
    Encodes protein sequences into numerical format using Label Encoding.
    :param sequences: A list of raw protein sequences (strings).
    :return: A list of encoded sequences and the classes (amino acids).
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(list("ACDEFGHIKLMNPQRSTVWYXZBJUO"))
    encoded_seqs = [torch.tensor(label_encoder.transform(list(seq))) for seq in sequences]
    return encoded_seqs, label_encoder.classes_


def pad_collate(batch):
    """
    Collate function that pads input sequences and target sequences separately.
    :param batch: A batch of tuples of (input_sequence, target_sequence).
    :return: Two tensors: padded inputs and padded targets.
    """
    # Separate the input and target sequences
    input_seqs, target_seqs = zip(*batch)

    # Pad the input and target sequences
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)

    return input_seqs_padded, target_seqs_padded


class LSTMProteinGenerator(nn.Module):
    """LSTM model for protein sequence generation."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        Initializes the LSTM model.
        :param vocab_size: The size of the vocabulary (number of unique amino acids).
        :param embedding_dim: The size of the embedding vector.
        :param hidden_dim: The size of the LSTM hidden layer.
        :param num_layers: The number of LSTM layers.
        """
        super(LSTMProteinGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass through the model.
        :param x: Input sequence of encoded amino acids.
        :return: The output logits from the model.
        """
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out)
        return out


def train(model, train_loader, val_loader, optimizer, criterion, epochs, model_path):
    """
    Trains the LSTM model with early stopping and gradient clipping.
    :param model: The LSTM model instance.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param optimizer: Optimizer for the model parameters.
    :param criterion: Loss function.
    :param epochs: Number of training epochs.
    :param model_path: Path to save the best model.
    """
    start_time = time.time()
    model.train()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch  # Assuming batch contains inputs and targets
            outputs = model(inputs)
            targets = targets.long()  # Convert targets to Long type
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1)  # Apply gradient clipping

            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch  # Ensure validation batch is unpacked similarly
                targets = targets.long()  # Ensure targets are Long type
                outputs = model(inputs)
                loss = criterion(outputs.transpose(1, 2), targets)
                val_loss += loss.item()

        scheduler.step(val_loss / len(val_loader))

        val_loss_avg = val_loss / len(val_loader)
        print(
            f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss_avg}')

        # Checkpointing and early stopping
        if val_loss_avg < best_val_loss:
            print(f"Validation loss improved from {best_val_loss} to {val_loss_avg}. Saving model...")
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break  # Break from the loop if patience limit is reached

    end_time = time.time()
    print(f"Training completed. Best model saved to {model_path}. Time taken: {end_time - start_time: .2f} seconds.")

def main(fasta_file, model_path):
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    encoded_seqs, _ = encode_sequences(sequences)
    vocab_size = len(set("ACDEFGHIKLMNPQRSTVWYXZBJUO")) + 1

    train_seqs, val_seqs = train_test_split(encoded_seqs, test_size=0.2)

    train_dataset = ProteinSequenceDataset(train_seqs)
    val_dataset = ProteinSequenceDataset(val_seqs)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, collate_fn=pad_collate)

    embedding_dim = 64
    hidden_dim = 128
    num_layers = 2

    model = LSTMProteinGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    train(model, train_loader, val_loader, optimizer, criterion, epochs, model_path)

if __name__ == "__main__":
    fasta_file = "../../../data/training data/uniprot_sprot.fasta"
    model_path = "../../models/rnn_model"

    main(fasta_file, model_path)
