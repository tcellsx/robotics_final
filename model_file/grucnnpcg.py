"""
Neural Drum Pattern Generator
Trains a GRU-CNN neural network model to generate new drum patterns based on the Groove dataset
With comprehensive improvements and without class weights
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D, Reshape, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import random
import argparse
import json
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Control GPU memory allocation if GPUs are available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected and configured: {len(gpus)}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


def create_model_dir(model_dir='model'):
    """
    Create a directory for storing models if it doesn't exist

    Args:
        model_dir: Directory name

    Returns:
        Path to the model directory
    """
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    else:
        print(f"Using existing model directory: {model_dir}")

    return model_dir


def load_data(file_path):
    """
    Load the drum pattern data from CSV

    Args:
        file_path: Path to the data file

    Returns:
        DataFrame containing the drum patterns
    """
    # Check if file exists in current directory if no path provided
    if not os.path.exists(file_path):
        # Try looking in current directory
        current_dir_path = os.path.join(os.getcwd(), os.path.basename(file_path))
        if os.path.exists(current_dir_path):
            file_path = current_dir_path
            print(f"Found input file in current directory: {file_path}")
        else:
            print(f"Error: Could not find input file {file_path}")
            return None

    # Load the CSV data
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def prepare_sequences(df, sequence_length=8, drum_types=None, cyclic_patterns=True):
    """
    Prepare sequences for training the neural network with improved handling
    of cyclic patterns

    Args:
        df: DataFrame containing the drum patterns
        sequence_length: Length of input sequences
        drum_types: List of drum types to include (if None, use all)
        cyclic_patterns: Whether to create sequences that cross pattern boundaries

    Returns:
        X: Input sequences
        y: Target values
        drum_types: List of drum types used
    """
    if 'time_step' in df.columns:
        # Data is in long format (each time step is a row)
        print("Processing data in long format (time steps as rows)")

        # If drum_types not specified, use all available except non-drum columns
        if drum_types is None:
            non_drum_cols = ['style', 'pattern_name', 'time_step']
            drum_types = [col for col in df.columns if col not in non_drum_cols]

        # Get all unique pattern names
        pattern_names = df['pattern_name'].unique()

        X = []
        y = []

        for pattern_name in pattern_names:
            # Get this pattern's data
            pattern_df = df[df['pattern_name'] == pattern_name].sort_values('time_step')

            # Extract just the drum hit data
            pattern_data = pattern_df[drum_types].values

            # Create sequences only within the same pattern (don't cross pattern boundaries)
            if len(pattern_data) > sequence_length:
                for i in range(len(pattern_data) - sequence_length):
                    X.append(pattern_data[i:i + sequence_length])
                    y.append(pattern_data[i + sequence_length])

                # Add cyclic pattern support
                if cyclic_patterns and len(pattern_data) >= 16:  # Assuming standard 16-step patterns
                    try:
                        # Create extended version of pattern to learn cyclic nature
                        extended_data = np.vstack([pattern_data, pattern_data[:sequence_length + 1]])

                        # Add sequences that cross pattern boundaries
                        boundary_start = len(pattern_data) - sequence_length + 1
                        for i in range(sequence_length - 1):
                            if boundary_start + i + sequence_length < len(extended_data):
                                seq_start = boundary_start + i
                                X.append(extended_data[seq_start:seq_start + sequence_length])
                                y.append(extended_data[seq_start + sequence_length])
                    except Exception as e:
                        print(f"Warning: Error creating cyclic patterns: {e}")

    else:
        # Data is in flat format (each pattern is a row)
        print("Processing data in flat format (patterns as rows)")

        # Determine drum types and pattern length
        if drum_types is None:
            # Find all column names that match drum_X format
            drum_cols = [col for col in df.columns if
                         '_' in col and col.split('_')[0] in ['kick', 'snare', 'hihat', 'tom1', 'tom2']]
            # Get unique drum types
            drum_types = sorted(list(set([col.split('_')[0] for col in drum_cols])))

        # Determine the pattern length from column names
        max_steps = 0
        for col in df.columns:
            if '_' in col:
                try:
                    step = int(col.split('_')[1])
                    max_steps = max(max_steps, step + 1)  # +1 because zero-indexed
                except ValueError:
                    continue

        X = []
        y = []

        # Process each pattern
        for _, row in df.iterrows():
            # Reconstruct the sequence for each drum type
            pattern_data = np.zeros((max_steps, len(drum_types)))

            for i, drum in enumerate(drum_types):
                for step in range(max_steps):
                    col = f"{drum}_{step}"
                    if col in df.columns:
                        pattern_data[step, i] = row[col]

            # Create sequences
            for i in range(max_steps - sequence_length):
                X.append(pattern_data[i:i + sequence_length])
                y.append(pattern_data[i + sequence_length])

            # Add cyclic support for flat format too
            if cyclic_patterns and max_steps >= 16:
                try:
                    extended_data = np.vstack([pattern_data, pattern_data[:sequence_length + 1]])
                    boundary_start = max_steps - sequence_length + 1
                    for i in range(sequence_length - 1):
                        if boundary_start + i + sequence_length < len(extended_data):
                            seq_start = boundary_start + i
                            X.append(extended_data[seq_start:seq_start + sequence_length])
                            y.append(extended_data[seq_start + sequence_length])
                except Exception as e:
                    print(f"Warning: Error creating cyclic patterns: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Created {len(X)} sequences with shape {X.shape}")
    return X, y, drum_types


def data_augmentation(X, y, noise_level=0.05, time_shift_prob=0.3):
    """
    Apply data augmentation to training data

    Args:
        X: Input sequences
        y: Target values
        noise_level: Probability of adding noise
        time_shift_prob: Probability of time shifting

    Returns:
        Augmented training data
    """
    X_aug = np.copy(X)
    y_aug = np.copy(y)

    # Add random noise
    for i in range(len(X_aug)):
        mask = np.random.random(X_aug[i].shape) < noise_level
        X_aug[i] = np.logical_xor(X_aug[i].astype(bool), mask).astype(float)

    # Apply time shifting to some samples
    shift_indices = np.random.random(len(X_aug)) < time_shift_prob
    for i in np.where(shift_indices)[0]:
        # For selected samples, shift left or right by 1-2 steps
        shift = np.random.choice([-2, -1, 1, 2])
        if shift > 0:  # Shift right
            X_aug[i] = np.roll(X_aug[i], shift=shift, axis=0)
            X_aug[i][:shift] = 0  # Clear beginning after shift
        else:  # Shift left
            X_aug[i] = np.roll(X_aug[i], shift=shift, axis=0)
            X_aug[i][shift:] = 0  # Clear end after shift

    return X_aug, y_aug


def build_gru_cnn_model(input_shape, output_shape, l2_factor=0.001, model_type='gru_cnn'):
    """
    Build a GRU-CNN hybrid model for drum pattern generation

    Args:
        input_shape: Shape of input sequences
        output_shape: Shape of output sequences
        l2_factor: L2 regularization factor
        model_type: Type of model architecture to use ('gru_cnn', 'cnn', 'gru', or 'transformer')

    Returns:
        Compiled Keras model
    """
    if model_type == 'gru_cnn':
        # GRU-CNN Hybrid Model
        inputs = Input(shape=input_shape)

        # Convolutional layers to extract local features
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_factor))(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_factor))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # GRU layer to capture sequential patterns
        x = GRU(160, return_sequences=False,
                kernel_regularizer=l2(l2_factor),
                recurrent_regularizer=l2(l2_factor / 2))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Output layers
        x = Dense(128, activation='relu', kernel_regularizer=l2(l2_factor))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(output_shape, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

    elif model_type == 'cnn':
        # CNN-only Model
        inputs = Input(shape=input_shape)

        # Convolutional block 1
        x = Conv1D(64, kernel_size=2, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_factor))(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(64, kernel_size=2, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_factor))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Dropout(0.3)(x)

        # Convolutional block 2
        x = Conv1D(128, kernel_size=2, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_factor))(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=2, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_factor))(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.3)(x)

        # Output layers
        x = Dense(128, activation='relu', kernel_regularizer=l2(l2_factor))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(output_shape, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

    elif model_type == 'gru':
        # GRU-only Model
        inputs = Input(shape=input_shape)

        # First GRU layer
        x = GRU(128, return_sequences=True,
                kernel_regularizer=l2(l2_factor),
                recurrent_regularizer=l2(l2_factor / 2))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Second GRU layer
        x = GRU(192, return_sequences=False,
                kernel_regularizer=l2(l2_factor),
                recurrent_regularizer=l2(l2_factor / 2))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Output layers
        x = Dense(128, activation='relu', kernel_regularizer=l2(l2_factor))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(output_shape, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Use Adam optimizer with learning rate
    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()
    return model


def train_model(X, y, model_dir='model', epochs=150, batch_size=32, use_augmentation=True, model_type='gru_cnn'):
    """
    Train the neural network with comprehensive improvements:
    - Learning rate scheduling
    - Data augmentation

    Args:
        X: Input sequences
        y: Target values
        model_dir: Directory to save the trained model
        epochs: Number of epochs to train
        batch_size: Batch size for training
        use_augmentation: Whether to use data augmentation
        model_type: Type of model architecture to use

    Returns:
        Trained model and training history
    """
    # Create model save path
    model_save_path = os.path.join(model_dir, f'drum_model_{model_type}.h5')

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up model
    model = build_gru_cnn_model(X_train.shape[1:], y_train.shape[1], model_type=model_type)

    # Create callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience
        verbose=1,
        restore_best_weights=True,
        min_delta=0.0005  # Add minimum improvement threshold
    )

    # Add learning rate scheduler with improved settings
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Larger reduction when triggered
        patience=8,  # Increased patience
        min_lr=0.00001,
        verbose=1,
        cooldown=2  # Add cooldown period
    )

    # Apply data augmentation if enabled
    if use_augmentation:
        print("Applying data augmentation...")
        augmented_X, augmented_y = data_augmentation(X_train, y_train)
        X_train_aug = np.concatenate([X_train, augmented_X])
        y_train_aug = np.concatenate([y_train, augmented_y])
        print(f"Training data increased from {len(X_train)} to {len(X_train_aug)} sequences")
    else:
        X_train_aug, y_train_aug = X_train, y_train

    # Train model with all improvements
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        shuffle=True
    )

    # Save training history plot
    plot_training_history(history, model_dir, model_type)

    return model, history


def plot_training_history(history, model_dir='model', model_type='gru_cnn'):
    """
    Plot the training history

    Args:
        history: Training history from model.fit()
        model_dir: Directory to save the plot
        model_type: Type of model architecture used
    """
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type.upper()} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_type.upper()} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(model_dir, f'training_history_{model_type}.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Training history plot saved to {plot_path}")


def generate_pattern(model, seed_sequence, length=16, drum_types=None, threshold=0.5, temperature=1.0):
    """
    Generate a new drum pattern with temperature control

    Args:
        model: Trained model
        seed_sequence: Initial sequence to start generation
        length: Length of pattern to generate
        drum_types: List of drum types
        threshold: Probability threshold for drum hits
        temperature: Controls randomness (lower=conservative, higher=more creative)

    Returns:
        Generated pattern
    """
    if drum_types is None:
        drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    # Create a copy of the seed sequence
    generated = seed_sequence.copy()

    # Wrap in try-except to handle potential errors
    try:
        for _ in range(length):
            # Use the last sequence_length steps to predict the next step
            sequence = generated[-seed_sequence.shape[0]:]
            sequence = np.reshape(sequence, (1, sequence.shape[0], sequence.shape[1]))

            # Predict the next step
            next_probs = model.predict(sequence, verbose=0)[0]

            # Apply temperature scaling
            if temperature != 1.0:
                # Convert to log odds (logits)
                # Clip probabilities to avoid log(0) or log(1)
                next_probs = np.clip(next_probs, 1e-7, 1 - 1e-7)
                logits = np.log(next_probs / (1 - next_probs))
                # Apply temperature
                scaled_logits = logits / temperature
                # Convert back to probabilities
                next_probs = 1 / (1 + np.exp(-scaled_logits))

            # Apply threshold
            next_step = (next_probs > threshold).astype(int)

            # Add to generated sequence
            generated = np.vstack([generated, next_step])
    except Exception as e:
        print(f"Error during pattern generation: {e}")
        import traceback
        traceback.print_exc()

        # Generate a fallback pattern if prediction fails
        remaining = length - (generated.shape[0] - seed_sequence.shape[0])
        if remaining > 0:
            fallback = np.zeros((remaining, len(drum_types)))
            # Add basic rhythm to fallback
            for i in range(remaining):
                if i % 4 == 0:  # Basic kick pattern
                    fallback[i, 0] = 1
                if i % 8 == 4:  # Basic snare pattern
                    fallback[i, 1] = 1
                if i % 2 == 0:  # Basic hihat pattern
                    fallback[i, 2] = 1
            generated = np.vstack([generated, fallback])

    return generated[seed_sequence.shape[0]:]


def print_drum_pattern(pattern, drum_types=None):
    """
    Print a drum pattern in ASCII format

    Args:
        pattern: Drum pattern to print
        drum_types: List of drum types
    """
    if drum_types is None:
        drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    # Print header
    print("    ", end="")
    for i in range(pattern.shape[0]):
        if i % 4 == 0:
            print("|", end="")
        else:
            print("-", end="")
    print("|")

    # Print each drum type
    for i, drum in enumerate(drum_types):
        if i < len(drum_types):  # Safety check
            print(f"{drum:5s}", end="")
            for step in range(pattern.shape[0]):
                if pattern[step, i] > 0:
                    print("X", end="")
                else:
                    print(".", end="")
            print()

    # Print footer
    print("    ", end="")
    for i in range(pattern.shape[0]):
        if i % 4 == 0:
            print("|", end="")
        else:
            print("-", end="")
    print("|")


def save_pattern_to_json(pattern, file_path, drum_types=None):
    """
    Save a generated pattern to JSON

    Args:
        pattern: Generated pattern
        file_path: Output file path
        drum_types: List of drum types
    """
    if drum_types is None:
        drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    # Create pattern dictionary
    pattern_dict = {}

    for i, drum in enumerate(drum_types):
        if i < len(drum_types) and i < pattern.shape[1]:  # Safety check
            pattern_dict[drum] = [int(hit) for hit in pattern[:, i]]

    # Save to JSON
    with open(file_path, 'w') as f:
        json.dump(pattern_dict, f, indent=2)

    print(f"Pattern saved to {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Train an improved drum pattern generator model')
    parser.add_argument('--input', default='drum_patterns.csv', help='Input CSV file path')
    parser.add_argument('--model-dir', default='model', help='Directory to save model and outputs')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--sequence-length', type=int, default=8, help='Length of input sequences')
    parser.add_argument('--pattern-length', type=int, default=16, help='Length of generated patterns')
    parser.add_argument('--l2-factor', type=float, default=0.002, help='L2 regularization factor')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature (randomness)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Generation threshold for drum hits')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--model-type', default='gru_cnn',
                        choices=['gru_cnn', 'cnn', 'gru'],
                        help='Type of model architecture to use')

    args = parser.parse_args()

    try:
        print(f"TensorFlow version: {tf.__version__}")

        # Create model directory
        model_dir = create_model_dir(args.model_dir)

        # Load data
        df = load_data(args.input)
        if df is None:
            return

        # Prepare sequences
        X, y, drum_types = prepare_sequences(df, sequence_length=args.sequence_length)

        # Verify data shapes
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Drum types: {drum_types}")

        # Check if model already exists
        model_path = os.path.join(model_dir, f'drum_model_{args.model_type}.h5')
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = load_model(model_path)
        else:
            # Train model
            print(f"Training new {args.model_type.upper()} model with {args.epochs} epochs")
            model, _ = train_model(
                X, y,
                model_dir=model_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_augmentation=not args.no_augmentation,
                model_type=args.model_type
            )

        # Generate patterns
        print(f"\nGenerating a new drum pattern with {args.model_type.upper()} model")

        # Select a random seed sequence
        seed_idx = random.choice(range(len(X)))
        seed_sequence = X[seed_idx]

        # Generate pattern
        generated = generate_pattern(
            model,
            seed_sequence,
            length=args.pattern_length,
            drum_types=drum_types,
            threshold=args.threshold,
            temperature=args.temperature
        )

        # Print pattern
        print("\nGenerated Pattern:")
        print_drum_pattern(generated, drum_types)

        # Save pattern as JSON
        output_file = f'generated_pattern_{args.model_type}.json'
        save_pattern_to_json(generated, output_file, drum_types)

        # Generate additional patterns with different temperatures if requested
        if args.temperature == 1.0:
            try:
                # Generate one cooler (more conservative) pattern
                cool_pattern = generate_pattern(
                    model, seed_sequence,
                    length=args.pattern_length,
                    drum_types=drum_types,
                    temperature=0.7
                )

                # Generate one hotter (more creative) pattern
                hot_pattern = generate_pattern(
                    model, seed_sequence,
                    length=args.pattern_length,
                    drum_types=drum_types,
                    temperature=1.3
                )

                # Print and save these patterns
                print("\nConservative Pattern (temperature=0.7):")
                print_drum_pattern(cool_pattern, drum_types)
                save_pattern_to_json(cool_pattern, f'conservative_pattern_{args.model_type}.json', drum_types)

                print("\nCreative Pattern (temperature=1.3):")
                print_drum_pattern(hot_pattern, drum_types)
                save_pattern_to_json(hot_pattern, f'creative_pattern_{args.model_type}.json', drum_types)
            except Exception as variant_error:
                print(f"Error generating variant patterns: {variant_error}")

        print("\nPattern generation completed successfully")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
#### 使用 GRU-CNN 混合模型（默认）
python gru_cnn_drum_generator.py --model-type gru_cnn

# 使用纯 CNN 模型
python gru_cnn_drum_generator.py --model-type cnn

# 使用纯 GRU 模型
python gru_cnn_drum_generator.py --model-type gru
"""
