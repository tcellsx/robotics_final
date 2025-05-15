"""
Neural Drum Pattern Generator using GAN (Generative Adversarial Network)
Trains a GAN model to generate new drum patterns based on the Groove dataset
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, Flatten, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, Conv1D, Conv1DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
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

def create_model_dir(model_dir='gan_model'):
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
    Prepare full patterns for the GAN model
    We train the GAN on full patterns rather than sequences

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
            drum_cols = [col for col in df.columns if '_' in col and col.split('_')[0] in ['kick', 'snare', 'hihat', 'tom1', 'tom2']]
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

def build_generator(latent_dim, pattern_shape, l2_factor=0.001):
    """
    Build the generator model for the GAN

    Args:
        latent_dim: Dimension of the latent space
        pattern_shape: Shape of the patterns to generate (timesteps, features)
        l2_factor: L2 regularization factor

    Returns:
        Generator model
    """
    model = Sequential(name='generator')

    # First layer processes the random noise vector
    model.add(Dense(128, kernel_regularizer=l2(l2_factor), input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    # Expand to the right shape for Conv1DTranspose
    model.add(Dense(pattern_shape[0] * 32, kernel_regularizer=l2(l2_factor)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    # Reshape for convolutional processing
    model.add(Reshape((pattern_shape[0], 32)))

    # Conv1DTranspose layers (like deconvolution) for temporal patterns
    model.add(Conv1D(32, kernel_size=3, padding='same', kernel_regularizer=l2(l2_factor)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Conv1D(16, kernel_size=3, padding='same', kernel_regularizer=l2(l2_factor)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    # Output layer with sigmoid activation for binary patterns
    model.add(Conv1D(pattern_shape[1], kernel_size=3, padding='same', activation='sigmoid'))

    # Print model summary
    model.summary()

    return model

def build_discriminator(pattern_shape, l2_factor=0.001):
    """
    Build the discriminator model for the GAN

    Args:
        pattern_shape: Shape of the patterns to discriminate (timesteps, features)
        l2_factor: L2 regularization factor

    Returns:
        Discriminator model
    """
    model = Sequential(name='discriminator')

    # First convolutional layer
    model.add(Conv1D(16, kernel_size=3, strides=1, padding='same',
                    kernel_regularizer=l2(l2_factor),
                    input_shape=pattern_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Second convolutional layer
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='same',
                    kernel_regularizer=l2(l2_factor)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Third convolutional layer
    model.add(Conv1D(64, kernel_size=3, strides=1, padding='same',
                    kernel_regularizer=l2(l2_factor)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=l2(l2_factor)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Print model summary
    model.summary()

    return model

def build_gan(generator, discriminator):
    """
    Build the combined GAN model

    Args:
        generator: Generator model
        discriminator: Discriminator model

    Returns:
        Combined GAN model
    """
    # Make the discriminator not trainable when training the generator
    discriminator.trainable = False

    # The GAN input is random noise and the output is the discriminator's evaluation
    gan_input = Input(shape=(generator.input_shape[1],))
    generator_output = generator(gan_input)
    gan_output = discriminator(generator_output)

    # Create and compile the combined model
    model = Model(gan_input, gan_output, name='gan')
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

    return model

def train_gan(patterns, latent_dim=100, epochs=150, batch_size=32, model_dir='gan_model',
              save_interval=10, use_augmentation=True):
    """
    Train the GAN model

    Args:
        patterns: Input patterns for training
        latent_dim: Dimension of the latent space
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_dir: Directory to save models and outputs
        save_interval: Interval (in epochs) to save generated patterns
        use_augmentation: Whether to use data augmentation

    Returns:
        Trained generator and discriminator models
    """
    # Apply data augmentation if enabled
    if use_augmentation:
        print("Applying data augmentation...")
        augmented_patterns = data_augmentation(patterns)
        patterns_train = np.concatenate([patterns, augmented_patterns])
        print(f"Training data increased from {len(patterns)} to {len(patterns_train)} patterns")
    else:
        patterns_train = patterns

    # Get pattern shape from training data
    pattern_shape = patterns_train.shape[1:]

    # Build models
    generator = build_generator(latent_dim, pattern_shape)
    discriminator = build_discriminator(pattern_shape)
    gan = build_gan(generator, discriminator)

    # Compile discriminator separately
    discriminator.compile(loss='binary_crossentropy',
                         optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                         metrics=['accuracy'])

    # Create arrays to store loss and accuracy history
    d_loss_history = []
    g_loss_history = []
    d_acc_history = []
    epoch_checkpoints = []

    # Training loop
    for epoch in range(epochs):
        # Train discriminator
        # Select a random batch of patterns
        idx = np.random.randint(0, patterns_train.shape[0], batch_size)
        real_patterns = patterns_train[idx]

        # Generate a batch of new patterns
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_patterns = generator.predict(noise, verbose=0)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_patterns, np.ones((batch_size, 1)) * 0.9)  # Label smoothing
        d_loss_fake = discriminator.train_on_batch(fake_patterns, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]*100:.2f}% | G Loss: {g_loss:.4f}")

        # Store history
        d_loss_history.append(d_loss[0])
        d_acc_history.append(d_loss[1])
        g_loss_history.append(g_loss)
        epoch_checkpoints.append(epoch + 1)

        # Save models periodically (but not intermediate patterns)
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            # Only save the latest model version, overwriting previous ones
            generator.save(os.path.join(model_dir, 'generator.h5'))
            discriminator.save(os.path.join(model_dir, 'discriminator.h5'))

    # Plot training history
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epoch_checkpoints, d_loss_history, label='Discriminator Loss')
    plt.plot(epoch_checkpoints, g_loss_history, label='Generator Loss')
    plt.title('Model Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epoch_checkpoints, d_acc_history, label='Discriminator Accuracy')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    history_path = os.path.join(model_dir, 'training_history.png')
    plt.savefig(history_path)
    plt.close()

    print(f"Training history saved to {history_path}")

    return generator, discriminator

def generate_patterns(generator, latent_dim, num_patterns=5, drum_types=None, threshold=0.5, temperature=1.0):
    """
    Generate drum patterns using the trained generator

    Args:
        generator: Trained generator model
        latent_dim: Dimension of the latent space
        num_patterns: Number of patterns to generate
        drum_types: List of drum types
        threshold: Threshold for binary output
        temperature: Controls randomness of generation

    Returns:
        List of generated patterns
    """
    if drum_types is None:
        drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    generated_patterns = []

    for i in range(num_patterns):
        # Generate random noise
        noise = np.random.normal(0, 1, (1, latent_dim))

        # Apply temperature to noise (higher temperature = more random)
        if temperature != 1.0:
            noise = noise * temperature

        # Generate pattern
        pattern = generator.predict(noise, verbose=0)[0]

        # Apply threshold for binary output
        binary_pattern = (pattern > threshold).astype(int)

        generated_patterns.append(binary_pattern)

    return generated_patterns

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
        if i < pattern.shape[1]:  # Safety check
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

def generate_latent_space_walk(generator, latent_dim, n_steps=10, drum_types=None):
    """
    Generate a series of patterns by walking through the latent space

    Args:
        generator: Trained generator model
        latent_dim: Dimension of the latent space
        n_steps: Number of steps in the walk
        drum_types: List of drum types

    Returns:
        Array of generated patterns
    """
    if drum_types is None:
        drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    # Generate starting and ending points
    z_start = np.random.normal(0, 1, (1, latent_dim))
    z_end = np.random.normal(0, 1, (1, latent_dim))

    # Generate patterns along the path
    patterns = []
    for alpha in np.linspace(0, 1, n_steps):
        z = z_start * (1 - alpha) + z_end * alpha
        pattern = generator.predict(z, verbose=0)[0]
        binary_pattern = (pattern > 0.5).astype(int)
        patterns.append(binary_pattern)

    return patterns

def main():
    parser = argparse.ArgumentParser(description='Train a GAN drum pattern generator')
    parser.add_argument('--input', default='drum_patterns.csv', help='Input CSV file path')
    parser.add_argument('--model-dir', default='gan_model', help='Directory to save model and outputs')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--pattern-length', type=int, default=16, help='Length of patterns')
    parser.add_argument('--latent-dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature (randomness)')
    parser.add_argument('--num-patterns', type=int, default=1, help='Number of patterns to generate')
    parser.add_argument('--save-interval', type=int, default=50, help='Interval to save models during training')

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

        # Check if generator model already exists
        generator_path = os.path.join(model_dir, 'generator.h5')

        if os.path.exists(generator_path):
            print(f"Loading existing generator from {generator_path}")
            generator = load_model(generator_path)
        else:
            # Train GAN
            print(f"Training new GAN with {args.epochs} epochs")
            generator, _ = train_gan(
                patterns,
                latent_dim=args.latent_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_dir=model_dir,
                save_interval=args.save_interval,
                use_augmentation=not args.no_augmentation
            )

        # Generate a single pattern (or the number specified by the user)
        print(f"\nGenerating {args.num_patterns} pattern(s) with temperature {args.temperature}")
        generated_patterns = generate_patterns(
            generator,
            args.latent_dim,
            num_patterns=args.num_patterns,
            drum_types=drum_types,
            temperature=args.temperature
        )

        # Only save one main pattern file (instead of multiple files)
        main_pattern = generated_patterns[0]  # Use the first generated pattern
        print("\nGenerated Drum Pattern:")
        print_drum_pattern(main_pattern, drum_types)
        save_pattern_to_json(main_pattern, 'generated_pattern.json', drum_types)

        # Print additional patterns if requested but don't save them
        for i in range(1, len(generated_patterns)):
            print(f"\nAdditional Pattern {i}:")
            print_drum_pattern(generated_patterns[i], drum_types)

        print("\nPattern generation completed successfully")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()