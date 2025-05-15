"""
Neural Drum Pattern Generator with Variational Autoencoder (VAE)
Trains a VAE model to generate new drum patterns based on the Groove dataset
Allows for interpolation between patterns and controlled generation
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
import argparse
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

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


def create_model_dir(model_dir='vae_model'):
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


def prepare_patterns(df, pattern_length=16, drum_types=None):
    """
    Prepare full patterns for the VAE model
    Unlike sequence preparation for LSTM, we train the VAE on full patterns

    Args:
        df: DataFrame containing the drum patterns
        pattern_length: Length of drum patterns to use
        drum_types: List of drum types to include (if None, use all)

    Returns:
        patterns: Array of drum patterns
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

        # List to store complete patterns
        patterns = []

        for pattern_name in pattern_names:
            # Get this pattern's data
            pattern_df = df[df['pattern_name'] == pattern_name].sort_values('time_step')

            # Extract just the drum hit data
            pattern_data = pattern_df[drum_types].values

            # Handle patterns shorter or longer than desired length
            if 0 < len(pattern_data) < pattern_length:
                # For shorter patterns, repeat to fill desired length
                repeats = int(np.ceil(pattern_length / len(pattern_data)))
                extended_pattern = np.tile(pattern_data, (repeats, 1))
                patterns.append(extended_pattern[:pattern_length])
            elif len(pattern_data) >= pattern_length:
                # For longer patterns, take the first pattern_length steps
                patterns.append(pattern_data[:pattern_length])

                # Also add additional segments if pattern is long enough
                if len(pattern_data) >= 2 * pattern_length:
                    segments = len(pattern_data) // pattern_length
                    for i in range(1, segments):
                        start_idx = i * pattern_length
                        end_idx = start_idx + pattern_length
                        if end_idx <= len(pattern_data):
                            patterns.append(pattern_data[start_idx:end_idx])

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

        # Set pattern length based on what's in the data if not specified
        if max_steps > 0 and pattern_length > max_steps:
            print(f"Warning: Requested pattern_length {pattern_length} exceeds available steps {max_steps}")
            pattern_length = max_steps

        patterns = []

        # Process each pattern
        for _, row in df.iterrows():
            # Reconstruct the sequence for each drum type
            pattern_data = np.zeros((pattern_length, len(drum_types)))

            for i, drum in enumerate(drum_types):
                for step in range(pattern_length):
                    col = f"{drum}_{step}"
                    if col in df.columns:
                        pattern_data[step, i] = row[col]

            patterns.append(pattern_data)

    # Convert to numpy array
    patterns_array = np.array(patterns)
    print(f"Created {len(patterns_array)} patterns with shape {patterns_array.shape}")
    return patterns_array, drum_types


def data_augmentation(patterns, noise_level=0.05, time_shift_prob=0.3):
    """
    Apply data augmentation to patterns

    Args:
        patterns: Input patterns
        noise_level: Probability of adding noise
        time_shift_prob: Probability of time shifting

    Returns:
        Augmented patterns
    """
    patterns_aug = np.copy(patterns)

    # Add random noise
    for i in range(len(patterns_aug)):
        mask = np.random.random(patterns_aug[i].shape) < noise_level
        patterns_aug[i] = np.logical_xor(patterns_aug[i].astype(bool), mask).astype(float)

    # Apply time shifting to some samples
    shift_indices = np.random.random(len(patterns_aug)) < time_shift_prob
    for i in np.where(shift_indices)[0]:
        # For selected samples, shift left or right by 1-4 steps
        shift = np.random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        patterns_aug[i] = np.roll(patterns_aug[i], shift=shift, axis=0)

    return patterns_aug


def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.

    Args:
        args: Mean and log of variance of Q(z|X)

    Returns:
        z: Sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # By default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_vae_model(input_shape, latent_dim=32, l2_factor=0.001):
    """
    Build a VAE model for drum pattern generation

    Args:
        input_shape: Shape of input patterns (timesteps, features)
        latent_dim: Dimension of latent space
        l2_factor: L2 regularization factor

    Returns:
        VAE model, encoder model, and decoder model
    """
    # Flatten input shape for VAE processing
    flattened_dim = input_shape[0] * input_shape[1]

    # Encoder
    inputs = Input(shape=input_shape, name='encoder_input')

    # Reshape input to (batch, timesteps*features) to process with dense layers
    x = Reshape((flattened_dim,))(inputs)

    # Encoder architecture
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # VAE latent space
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

    # Decoder architecture - mirror of encoder
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_factor))(latent_inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(flattened_dim, activation='sigmoid')(x)

    # Reshape back to original dimensions
    outputs = Reshape(input_shape)(outputs)

    # Instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # Instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # VAE loss
    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        K.flatten(inputs), K.flatten(outputs)
    )
    reconstruction_loss *= flattened_dim

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Use Adam optimizer - Note: We don't use metrics with VAE
    optimizer = Adam(learning_rate=0.001)
    vae.compile(optimizer=optimizer)

    return vae, encoder, decoder


def train_vae_model(patterns, model_dir='vae_model', epochs=150, batch_size=32, latent_dim=32, use_augmentation=True):
    """
    Train the VAE model

    Args:
        patterns: Input patterns
        model_dir: Directory to save the trained model
        epochs: Number of epochs to train
        batch_size: Batch size for training
        latent_dim: Dimension of latent space
        use_augmentation: Whether to use data augmentation

    Returns:
        Trained VAE model, encoder, decoder, and training history
    """
    # Create model save paths
    vae_path = os.path.join(model_dir, 'vae_model.h5')
    encoder_path = os.path.join(model_dir, 'encoder_model.h5')
    decoder_path = os.path.join(model_dir, 'decoder_model.h5')

    # Split data into training and validation sets
    patterns_train, patterns_val = train_test_split(patterns, test_size=0.2, random_state=42)

    # Build VAE model
    vae, encoder, decoder = build_vae_model(
        input_shape=patterns.shape[1:],
        latent_dim=latent_dim
    )

    # Create callbacks
    checkpoint = ModelCheckpoint(
        vae_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=True,
        min_delta=0.0005
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=0.00001,
        verbose=1,
        cooldown=2
    )

    # Apply data augmentation if enabled
    if use_augmentation:
        print("Applying data augmentation...")
        augmented_patterns = data_augmentation(patterns_train)
        patterns_train_aug = np.concatenate([patterns_train, augmented_patterns])
        print(f"Training data increased from {len(patterns_train)} to {len(patterns_train_aug)} patterns")
    else:
        patterns_train_aug = patterns_train

    # Train model
    history = vae.fit(
        patterns_train_aug, None,  # VAE doesn't need explicit target values
        validation_data=(patterns_val, None),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        shuffle=True
    )

    # Save encoder and decoder models
    encoder.save(encoder_path)
    decoder.save(decoder_path)

    # Save training history plot
    plot_training_history(history, model_dir)

    return vae, encoder, decoder, history


def plot_training_history(history, model_dir='vae_model'):
    """
    Plot the training history

    Args:
        history: Training history from model.fit()
        model_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Plot loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(model_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Training history plot saved to {plot_path}")


def generate_pattern_from_latent(decoder, z_vector, drum_types=None):
    """
    Generate a drum pattern from a latent vector

    Args:
        decoder: Decoder model
        z_vector: Latent vector
        drum_types: List of drum types

    Returns:
        Generated pattern
    """
    if drum_types is None:
        drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    # Expand dimensions for batch
    z_vector = np.expand_dims(z_vector, axis=0)

    # Generate pattern from latent vector
    generated = decoder.predict(z_vector, verbose=0)[0]

    # Apply threshold for binary output
    generated = (generated > 0.5).astype(int)

    return generated


def generate_random_pattern(decoder, latent_dim=32, drum_types=None):
    """
    Generate a random drum pattern by sampling from latent space

    Args:
        decoder: Decoder model
        latent_dim: Dimension of latent space
        drum_types: List of drum types

    Returns:
        Generated pattern
    """
    # Sample from standard normal distribution
    z_vector = np.random.normal(size=latent_dim)

    # Generate pattern
    return generate_pattern_from_latent(decoder, z_vector, drum_types)


def generate_interpolated_patterns(decoder, patterns, encoder, n_steps=5, drum_types=None):
    """
    Generate patterns by interpolating between two existing patterns

    Args:
        decoder: Decoder model
        patterns: Array of patterns
        encoder: Encoder model
        n_steps: Number of interpolation steps
        drum_types: List of drum types

    Returns:
        Array of interpolated patterns
    """
    if drum_types is None:
        drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    # Select two random patterns
    idx1, idx2 = random.sample(range(len(patterns)), 2)
    pattern1 = patterns[idx1:idx1 + 1]
    pattern2 = patterns[idx2:idx2 + 1]

    # Encode patterns to latent space
    z_mean1, _, _ = encoder.predict(pattern1, verbose=0)
    z_mean2, _, _ = encoder.predict(pattern2, verbose=0)

    # Create interpolation steps
    interpolated = []
    for alpha in np.linspace(0, 1, n_steps):
        z_interp = z_mean1 * (1 - alpha) + z_mean2 * alpha
        pattern_interp = generate_pattern_from_latent(decoder, z_interp[0], drum_types)
        interpolated.append(pattern_interp)

    return np.array(interpolated)


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


def save_latent_space_visualization(encoder, patterns, model_dir='vae_model'):
    """
    Create and save a visualization of the latent space

    Args:
        encoder: Encoder model
        patterns: Array of patterns
        model_dir: Directory to save the visualization
    """
    # Sample patterns
    if len(patterns) > 1000:
        indices = np.random.choice(len(patterns), 1000, replace=False)
        sample_patterns = patterns[indices]
    else:
        sample_patterns = patterns

    # Encode to latent space
    z_mean, _, _ = encoder.predict(sample_patterns, verbose=0)

    # For 2D visualization, take first two dimensions
    plt.figure(figsize=(10, 8))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5)
    plt.title('VAE Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)

    # Save visualization
    vis_path = os.path.join(model_dir, 'latent_space.png')
    plt.savefig(vis_path)
    plt.close()

    print(f"Latent space visualization saved to {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Train a VAE drum pattern generator model')
    parser.add_argument('--input', default='drum_patterns.csv', help='Input CSV file path')
    parser.add_argument('--model-dir', default='vae_model', help='Directory to save model and outputs')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--pattern-length', type=int, default=16, help='Length of patterns')
    parser.add_argument('--latent-dim', type=int, default=32, help='Dimension of latent space')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--interpolation-steps', type=int, default=5, help='Number of interpolation steps')

    args = parser.parse_args()

    try:
        print(f"TensorFlow version: {tf.__version__}")

        # Create model directory
        model_dir = create_model_dir(args.model_dir)

        # Load data
        df = load_data(args.input)
        if df is None:
            return

        # Prepare patterns
        patterns, drum_types = prepare_patterns(df, pattern_length=args.pattern_length)

        # Verify data shapes
        print(f"Patterns shape: {patterns.shape}")
        print(f"Drum types: {drum_types}")

        # Check if model already exists
        vae_path = os.path.join(model_dir, 'vae_model.h5')
        encoder_path = os.path.join(model_dir, 'encoder_model.h5')
        decoder_path = os.path.join(model_dir, 'decoder_model.h5')

        if os.path.exists(vae_path) and os.path.exists(encoder_path) and os.path.exists(decoder_path):
            print(f"Loading existing models from {model_dir}")
            vae = load_model(vae_path, compile=False)
            encoder = load_model(encoder_path, compile=False)
            decoder = load_model(decoder_path, compile=False)
        else:
            # Train model
            print(f"Training new VAE model with {args.epochs} epochs")
            vae, encoder, decoder, history = train_vae_model(
                patterns,
                model_dir=model_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                latent_dim=args.latent_dim,
                use_augmentation=not args.no_augmentation
            )

        # Create latent space visualization
        print("Creating latent space visualization...")
        save_latent_space_visualization(encoder, patterns, model_dir)

        # Generate random pattern
        print("\nGenerating random pattern from latent space")
        random_pattern = generate_random_pattern(decoder, args.latent_dim, drum_types)
        print("\nRandom Generated Pattern:")
        print_drum_pattern(random_pattern, drum_types)
        save_pattern_to_json(random_pattern, 'vae_random_pattern.json', drum_types)

        # Generate interpolated patterns
        print(f"\nGenerating {args.interpolation_steps} interpolated patterns")
        interpolated_patterns = generate_interpolated_patterns(
            decoder, patterns, encoder,
            n_steps=args.interpolation_steps,
            drum_types=drum_types
        )

        for i, pattern in enumerate(interpolated_patterns):
            print(f"\nInterpolated Pattern {i + 1}/{args.interpolation_steps}:")
            print_drum_pattern(pattern, drum_types)
            save_pattern_to_json(pattern, f'vae_interpolated_{i + 1}.json', drum_types)

        print("\nPattern generation completed successfully")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()