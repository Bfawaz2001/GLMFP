import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from Bio import SeqIO
import time
from tensorflow.keras.callbacks import ModelCheckpoint


def create_rnn_model(input_dim, output_dim, embedding_dim=64, rnn_units=128):
    """
    Creates an RNN model for processing sequences.
    """
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim),
        SimpleRNN(rnn_units, return_sequences=False),
        Dense(output_dim, activation='softmax')
    ])
    return model


def preprocess_sequences(sequences, label_encoder, max_length=None):
    """
    Encodes and pads sequences for model input.
    """
    sequences_int = [label_encoder.transform(list(seq)) for seq in sequences]
    sequences_padded = pad_sequences(sequences_int, maxlen=max_length, padding='post')
    return sequences_padded


def load_and_preprocess_data(filepath, label_encoder, test_size=0.2, random_state=42):
    """
    Loads and preprocesses sequence data from a FASTA file.
    """
    sequences = [str(record.seq) for record in SeqIO.parse(filepath, "fasta")]
    sequences_padded = preprocess_sequences(sequences, label_encoder)
    X_train, X_test = train_test_split(sequences_padded, test_size=test_size, random_state=random_state)
    return np.array(X_train), np.array(X_test)


def main(fasta_file):
    """
    Main execution function for loading data, training, and evaluating the model.
    """
    print("Starting the script...")

    amino_acids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    embedding_dim = 64
    rnn_units = 128
    batch_size = 32
    epochs = 10

    print("Initializing the Label Encoder...")
    label_encoder = LabelEncoder()
    label_encoder.fit(list(amino_acids))

    print("Loading and preprocessing data...")
    X_train, X_test = load_and_preprocess_data(fasta_file, label_encoder)

    print("Creating the RNN model...")
    model = create_rnn_model(input_dim=len(amino_acids), output_dim=len(amino_acids),
                             embedding_dim=embedding_dim, rnn_units=rnn_units)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(
        'protein_rnn_model_checkpoint.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    print("Training the model...")
    start_time = time.time()
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, X_test), callbacks=[checkpoint_callback])
    end_time = time.time()

    print("Evaluating the model...")
    loss, accuracy = model.evaluate(X_test, X_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    model.save('protein_rnn_model_final.h5')
    print("Model training and evaluation complete.")
    print("Time to build model: ", end_time - start_time)


if __name__ == "__main__":
    fasta_file = '../../../data/training data/uniprot_sprot.fasta'  # Ensure this path points to your FASTA file
    main(fasta_file)