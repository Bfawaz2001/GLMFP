import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from Bio import SeqIO
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Ensure TensorFlow is using the M2 chip efficiently
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


def preprocess_sequences(fasta_file):
    """
    Load sequences from a FASTA file and preprocess them for the LSTM model.
    This function encodes sequences and pads them to a uniform length.

    Args:
    - fasta_file: Path to the FASTA file containing protein sequences.

    Returns:
    - encoded_sequences: Encoded and padded sequences as a numpy array.
    - label_encoder: The LabelEncoder instance used for encoding sequences.
    """
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))

    # Encode sequences
    label_encoder = LabelEncoder()
    label_encoder.fit(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))  # Fit encoder to the amino acid alphabet
    encoded_sequences = [label_encoder.transform(list(seq)) for seq in sequences]

    # Find max sequence length and pad sequences
    max_length = max(len(seq) for seq in encoded_sequences)
    padded_sequences = pad_sequences(encoded_sequences, maxlen=max_length, padding='post')

    return np.array(padded_sequences), label_encoder


def build_lstm_model(input_dim, output_dim, embedding_dim=64, lstm_units=256):
    """
    Build an LSTM model optimized for the M2 chip.

    Args:
    - input_dim: The size of the input dimension (vocabulary size).
    - output_dim: The size of the output dimension (same as input_dim).
    - embedding_dim: Dimensionality of the embedding layer.
    - lstm_units: The number of units in the LSTM layer.

    Returns:
    - A compiled LSTM model.
    """
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=None),
        LSTM(lstm_units, return_sequences=False),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(fasta_file):
    """
    Main execution function to train and evaluate the LSTM model on protein sequences.

    Args:
    - fasta_file: Path to the FASTA file containing protein sequences.
    """
    print("Preprocessing sequences...")
    sequences, label_encoder = preprocess_sequences(fasta_file)
    vocab_size = len(label_encoder.classes_)

    # Split data into training and test sets
    X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)

    # Targets for training are the input sequences themselves, shifted by one position
    y_train = np.hstack([X_train[:, 1:], np.zeros((X_train.shape[0], 1))])
    y_test = np.hstack([X_test[:, 1:], np.zeros((X_test.shape[0], 1))])

    print("Building the LSTM model...")
    model = build_lstm_model(input_dim=vocab_size, output_dim=vocab_size, lstm_units=256)

    # ModelCheckpoint to save the best model
    checkpoint_path = "best_model.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    print("Training the model...")
    start_time = time.time()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpoint], batch_size=64)
    end_time = time.time()

    print(f"Model trained. Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    fasta_file = '../../../data/training data/uniprot_sprot.fasta'  # Update this path to your actual FASTA file location
    main(fasta_file)
