"""
Groove MIDI Dataset Download and Conversion Tool (No NumPy dependency)
Automatically downloads Google Magenta's Groove MIDI dataset and converts it to CSV format suitable for neural network training
Usage: python midi2csv.py --output drum_patterns.csv
"""

import os
import csv
import mido
import argparse
import urllib.request
import zipfile
import glob
from collections import defaultdict
import shutil
import random
import time

# MIDI drum note mapping (GM standard)
DRUM_MAP = {
    36: 'kick',  # Bass Drum
    35: 'kick',  # Acoustic Bass Drum
    38: 'snare',  # Acoustic Snare
    40: 'snare',  # Electric Snare
    37: 'snare',  # Side Stick
    42: 'hihat',  # Closed Hi-hat
    46: 'hihat',  # Open Hi-hat
    44: 'hihat',  # Pedal Hi-hat
    45: 'tom1',  # Low Tom
    48: 'tom1',  # Hi-Mid Tom
    47: 'tom1',  # Low-Mid Tom
    50: 'tom2',  # High Tom
    43: 'tom2',  # High Floor Tom
    41: 'tom2'  # Low Floor Tom
}


# Class for displaying download progress
class DownloadProgressBar:
    def __init__(self, total_size, desc="Downloading"):
        self.total_size = total_size
        self.downloaded = 0
        self.start_time = time.time()
        self.desc = desc
        self.last_print_time = 0
        self._print_progress()

    def update(self, chunk_size):
        self.downloaded += chunk_size
        # Limit progress bar refresh rate to avoid excessive output
        current_time = time.time()
        if current_time - self.last_print_time > 0.1:  # Update every 0.1 seconds
            self._print_progress()
            self.last_print_time = current_time

    def _print_progress(self):
        percent = min(100, self.downloaded * 100 / self.total_size)
        elapsed_time = time.time() - self.start_time

        # Calculate download speed (MB/s)
        speed = self.downloaded / (1024 * 1024 * elapsed_time) if elapsed_time > 0 else 0

        # Calculate remaining time
        if speed > 0 and self.downloaded < self.total_size:
            eta = (self.total_size - self.downloaded) / (speed * 1024 * 1024)
        else:
            eta = 0

        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        # Format sizes
        downloaded_mb = self.downloaded / (1024 * 1024)
        total_mb = self.total_size / (1024 * 1024)

        # Clear current line and print new progress
        print(
            f"\r{self.desc}: [{bar}] {percent:.1f}% {downloaded_mb:.1f}MB/{total_mb:.1f}MB {speed:.1f}MB/s ETA: {eta:.0f}s",
            end='', flush=True)

        if self.downloaded >= self.total_size:
            print("  Complete!")  # New line after completion


def download_groove_dataset(download_dir=None):
    """
    Download the Groove MIDI dataset

    Args:
        download_dir: Download directory, if None use current directory

    Returns:
        Dataset directory path
    """
    # If no download directory specified, use current directory
    if download_dir is None:
        download_dir = os.getcwd()  # Use current directory
    else:
        os.makedirs(download_dir, exist_ok=True)

    print(f"Download directory: {download_dir}")

    # Dataset URL
    dataset_url = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip"

    # Download path
    zip_path = os.path.join(download_dir, "groove-v1.0.0.zip")

    # Download dataset
    if not os.path.exists(zip_path):
        print(f"Preparing to download Groove MIDI dataset...")

        # Get file size
        with urllib.request.urlopen(dataset_url) as response:
            file_size = int(response.info().get('Content-Length', 0))

        # Create progress bar
        progress_bar = DownloadProgressBar(file_size, desc="Downloading Groove dataset")

        # Custom download callback function
        def report_progress(block_num, block_size, total_size):
            progress_bar.update(block_size)

        # Download file
        urllib.request.urlretrieve(dataset_url, zip_path, reporthook=report_progress)
        print(f"\nDownload complete: {zip_path}")
    else:
        print(f"Using already downloaded dataset: {zip_path}")

    # Extract dataset
    dataset_dir = os.path.join(download_dir, "groove")
    if not os.path.exists(dataset_dir):
        print(f"Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total file count
            total_files = len(zip_ref.infolist())
            print(f"Total files: {total_files}")

            # Create progress bar
            progress_bar = DownloadProgressBar(total_files, desc="Extracting dataset")

            # Extract files one by one and update progress
            for i, file in enumerate(zip_ref.infolist()):
                zip_ref.extract(file, download_dir)
                progress_bar.update(1)

        print(f"\nExtraction complete: {dataset_dir}")
    else:
        print(f"Using already extracted dataset: {dataset_dir}")

    return dataset_dir


def midi_to_drum_pattern(midi_file, resolution=16, bars=1):
    """
    Convert a MIDI file to a drum pattern

    Args:
        midi_file: Path to MIDI file
        resolution: Quantization resolution, default is 16 steps/bar
        bars: Number of bars to extract

    Returns:
        Dictionary with drum types as keys and binary sequences as values
    """
    try:
        # Load MIDI file
        midi = mido.MidiFile(midi_file)

        # Get MIDI file ticks per beat
        ticks_per_beat = midi.ticks_per_beat

        # Calculate ticks per bar (assuming 4/4 time)
        ticks_per_bar = ticks_per_beat * 4

        # Calculate quantization step size
        ticks_per_step = ticks_per_bar / resolution

        # Range to extract
        total_ticks = bars * ticks_per_bar

        # Initialize drum events collection
        drum_events = defaultdict(set)

        # Process all tracks
        for track in midi.tracks:
            current_tick = 0
            for msg in track:
                # Add delta time
                current_tick += msg.time

                # Only process note-on events
                if current_tick < total_ticks and msg.type == 'note_on' and msg.velocity > 0:
                    # Check if it's a drum note
                    if hasattr(msg, 'channel') and msg.channel == 9 or hasattr(msg, 'is_drum') and msg.is_drum:
                        # Calculate quantized step
                        step = int(round(current_tick / ticks_per_step))

                        # Get drum type
                        drum_type = DRUM_MAP.get(msg.note)
                        if drum_type:
                            drum_events[drum_type].add(step)

        # Create output pattern
        pattern = {}
        for drum in ['kick', 'snare', 'hihat', 'tom1', 'tom2']:
            # Create binary sequence
            pattern[drum] = [1 if i in drum_events[drum] else 0 for i in range(resolution * bars)]

        return pattern

    except Exception as e:
        print(f"Error processing MIDI file {midi_file}: {e}")
        return {drum: [0] * (resolution * bars) for drum in ['kick', 'snare', 'hihat', 'tom1', 'tom2']}


def is_valid_pattern(pattern, min_hits=2):
    """
    Check if a pattern is valid (not empty and has at least some drum hits)

    Args:
        pattern: Drum pattern dictionary
        min_hits: Minimum number of drum hits

    Returns:
        True if pattern is valid, False otherwise
    """
    if not pattern:
        return False

    # Count total drum hits
    total_hits = sum(sum(seq) for seq in pattern.values())

    # Check if there are enough hits
    return total_hits >= min_hits


def infer_style_from_path(file_path):
    """
    Infer style from file path

    Args:
        file_path: MIDI file path

    Returns:
        Inferred style name
    """
    # Common style keywords
    style_keywords = {
        'rock': ['rock', 'hard', 'punk', 'alt'],
        'jazz': ['jazz', 'swing', 'bebop', 'blues'],
        'funk': ['funk', 'disco', 'groove'],
        'hiphop': ['hip', 'hop', 'rap', 'trap'],
        'metal': ['metal', 'heavy', 'thrash'],
        'latin': ['latin', 'bossa', 'samba', 'salsa'],
        'pop': ['pop', 'dance', 'electro'],
        'country': ['country', 'folk'],
        'reggae': ['reggae', 'dub']
    }

    # Get file name and directory names
    file_name = os.path.basename(file_path).lower()
    dir_name = os.path.basename(os.path.dirname(file_path)).lower()
    parent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path))).lower()

    # Check keywords
    for style, keywords in style_keywords.items():
        for keyword in keywords:
            if keyword in file_name or keyword in dir_name or keyword in parent_dir:
                return style

    # Check special cases - Groove dataset style markers
    if 'groove' in dir_name or 'groove' in parent_dir:
        # Check player style
        for player_dir in ['session', 'drummer']:
            if player_dir in file_path:
                return 'groove'

    # If style cannot be inferred, return default
    return 'unknown'


def process_groove_dataset(dataset_dir, resolution=16, max_patterns_per_style=20, specific_styles=None,
                           max_total_patterns=None):
    """
    Process all MIDI files in the Groove dataset

    Args:
        dataset_dir: Dataset directory path
        resolution: Quantization resolution
        max_patterns_per_style: Maximum number of patterns to extract per style
        specific_styles: Only process specified styles (if None, process all)
        max_total_patterns: Total maximum number of patterns to extract (if None, no limit)

    Returns:
        Dictionary of patterns, with pattern names as keys and patterns as values
    """
    print(f"Starting to process Groove MIDI dataset...")

    # Dictionary to store patterns by style
    style_patterns = defaultdict(list)

    # Find all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi']:
        for file_path in glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True):
            midi_files.append(file_path)

    print(f"Found {len(midi_files)} MIDI files")

    # Shuffle files for more diverse patterns
    random.shuffle(midi_files)

    # Track total pattern count
    total_patterns_count = 0

    # Create progress bar
    print("Processing MIDI files...")
    progress_bar = DownloadProgressBar(len(midi_files), desc="Processing MIDI files")
    processed_count = 0

    # Process each file
    for midi_file in midi_files:
        # Update progress bar
        processed_count += 1
        progress_bar.update(1)

        # Infer style
        style = infer_style_from_path(midi_file)

        # Skip if specific styles are specified and current style is not in the list
        if specific_styles and style not in specific_styles:
            continue

        # Skip if maximum patterns for this style have been reached
        if len(style_patterns[style]) >= max_patterns_per_style:
            continue

        # Stop processing if maximum total patterns have been reached
        if max_total_patterns is not None and total_patterns_count >= max_total_patterns:
            print(f"\nReached maximum total pattern limit ({max_total_patterns}), stopping processing")
            break

        # Extract pattern
        pattern = midi_to_drum_pattern(midi_file, resolution)

        # Check if pattern is valid
        if is_valid_pattern(pattern):
            style_patterns[style].append(pattern)
            total_patterns_count += 1

    # Create output dictionary
    patterns = {}
    for style, style_pattern_list in style_patterns.items():
        # Add patterns for each style
        for i, pattern in enumerate(style_pattern_list):
            if i == 0:
                # First pattern uses style name
                patterns[style] = pattern
            else:
                # Subsequent patterns add a number
                patterns[f"{style}_{i + 1}"] = pattern

    print(f"\nSuccessfully extracted {len(patterns)} drum patterns across {len(style_patterns)} styles")

    return patterns


def print_pattern(pattern, title=None):
    """
    Print drum pattern as ASCII art

    Args:
        pattern: Drum pattern dictionary
        title: Optional title
    """
    if not pattern:
        print("Empty pattern")
        return

    if title:
        print(f"\n{title}")

    # Get pattern length
    length = len(next(iter(pattern.values())))

    # Print ruler
    print("    ", end="")
    for i in range(length):
        if i % 4 == 0:
            print("|", end="")
        else:
            print("-", end="")
    print("|")

    # Print pattern for each drum
    for drum in ['kick', 'snare', 'hihat', 'tom1', 'tom2']:
        if drum in pattern:
            print(f"{drum:5s}", end="")
            for i in range(length):
                if pattern[drum][i] > 0:
                    print("X", end="")
                else:
                    print(".", end="")
            print()

    # Print ruler
    print("    ", end="")
    for i in range(length):
        if i % 4 == 0:
            print("|", end="")
        else:
            print("-", end="")
    print("|")


def patterns_to_csv(patterns, output_path):
    """
    Save drum patterns to CSV format

    Args:
        patterns: Dictionary of drum patterns, keys are pattern names, values are drum data
        output_path: Output CSV file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Get all drum types
    drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    # Get pattern length
    pattern_length = len(next(iter(patterns.values()))['kick'])

    # Build CSV field names
    # Format: style, pattern_name, time_step, kick, snare, hihat, tom1, tom2
    fieldnames = ['style', 'pattern_name', 'time_step'] + drum_types

    # Open CSV file and write
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through all patterns
        for pattern_name, pattern in patterns.items():
            # Extract style from pattern name
            if '_' in pattern_name:
                style = pattern_name.split('_')[0]
            else:
                style = pattern_name

            # Iterate through each time step
            for time_step in range(pattern_length):
                row = {
                    'style': style,
                    'pattern_name': pattern_name,
                    'time_step': time_step
                }

                # Add value for each drum at this time step
                for drum in drum_types:
                    row[drum] = pattern[drum][time_step]

                # Write row
                writer.writerow(row)

    print(f"Saved drum patterns to {output_path}")

    # Show CSV file statistics
    num_patterns = len(patterns)
    num_rows = num_patterns * pattern_length
    print(f"CSV contains {num_patterns} drum patterns with {num_rows} rows")
    print(f"Each row contains {len(fieldnames)} fields: {', '.join(fieldnames)}")


def patterns_to_flat_csv(patterns, output_path):
    """
    Save drum patterns to a flattened CSV format, one row per pattern

    Args:
        patterns: Dictionary of drum patterns, keys are pattern names, values are drum data
        output_path: Output CSV file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Get all drum types
    drum_types = ['kick', 'snare', 'hihat', 'tom1', 'tom2']

    # Get pattern length
    pattern_length = len(next(iter(patterns.values()))['kick'])

    # Build CSV field names
    # Format: style, pattern_name, kick_0, kick_1, ..., snare_0, snare_1, ..., etc.
    fieldnames = ['style', 'pattern_name']
    for drum in drum_types:
        for i in range(pattern_length):
            fieldnames.append(f"{drum}_{i}")

    # Open CSV file and write
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through all patterns
        for pattern_name, pattern in patterns.items():
            # Extract style from pattern name
            if '_' in pattern_name:
                style = pattern_name.split('_')[0]
            else:
                style = pattern_name

            # Create row
            row = {
                'style': style,
                'pattern_name': pattern_name
            }

            # Add each step for each drum to the row
            for drum in drum_types:
                for i, value in enumerate(pattern[drum]):
                    row[f"{drum}_{i}"] = value

            # Write row
            writer.writerow(row)

    print(f"Saved drum patterns to {output_path}")

    # Show CSV file statistics
    num_patterns = len(patterns)
    print(f"CSV contains {num_patterns} drum patterns, one pattern per row")
    print(
        f"Each row contains {len(fieldnames)} fields, including style, pattern name, and {pattern_length * len(drum_types)} drum features")


def main():
    parser = argparse.ArgumentParser(
        description='Download and convert Groove MIDI dataset to CSV format for neural network training')

    # Add arguments
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--download-dir', help='Download directory path (optional, defaults to current directory)')
    parser.add_argument('--keep-downloads', action='store_true', help='Keep downloaded files (default: delete)')
    parser.add_argument('--resolution', type=int, default=16, help='Quantization resolution, default is 16 steps/bar')
    parser.add_argument('--max-patterns', type=int, default=20, help='Maximum patterns to extract per style')
    parser.add_argument('--max-total', type=int, help='Total maximum number of patterns to extract')
    parser.add_argument('--styles', nargs='+', help='Only process specified styles (e.g., rock jazz funk)')
    parser.add_argument('--preview', action='store_true', help='Show preview of extracted patterns')
    parser.add_argument('--flat', action='store_true', help='Use flattened format (one pattern per row)')
    parser.add_argument('--one-hot', action='store_true', help='Use one-hot encoding (0/1 becomes vector)')

    args = parser.parse_args()

    try:
        # Download dataset
        dataset_dir = download_groove_dataset(args.download_dir)

        # Process dataset
        patterns = process_groove_dataset(
            dataset_dir,
            resolution=args.resolution,
            max_patterns_per_style=args.max_patterns,
            specific_styles=args.styles,
            max_total_patterns=args.max_total
        )

        # Show preview
        if args.preview:
            # Select styles to preview
            preview_styles = args.styles if args.styles else ['rock', 'funk', 'jazz', 'hiphop', 'metal']
            shown_count = 0

            for style in preview_styles:
                if style in patterns:
                    print_pattern(patterns[style], f"{style.capitalize()} style")
                    shown_count += 1

                # Show at most 5 patterns
                if shown_count >= 5:
                    break

        # Save to CSV
        output_path = os.path.abspath(args.output)

        # Ensure file extension is .csv
        if not output_path.lower().endswith('.csv'):
            output_path += '.csv'

        # Choose CSV format based on arguments
        if args.flat:
            patterns_to_flat_csv(patterns, output_path)
        else:
            patterns_to_csv(patterns, output_path)

        print(
            f"\nFor further data preprocessing for neural network training, you can use Python's pandas and scikit-learn libraries to process the CSV file.")

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()