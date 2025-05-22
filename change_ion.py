#!/usr/bin/env python3
"""
Auto Drum Pattern Updater - Simplified Version (Custom Directory Structure)

Automatically reads generated_pattern.json file in the current directory,
and updates the hardcoded drum patterns in the sketch_apr24a.ino file
located in the sketch_apr24a subfolder.
"""

import json
import re
import os
from pathlib import Path

# Configuration - Modify as needed
DEFAULT_JSON_FILE = "generated_pattern.json"  # Default JSON filename
DEFAULT_ARDUINO_FOLDER = "sketch_apr24a"  # Arduino code folder
DEFAULT_ARDUINO_FILE = "sketch_apr24a.ino"  # Default Arduino filename
MAKE_BACKUP = True  # Create backup files


def read_json_file(json_path):
    """
    Read JSON file and return drum patterns
    """
    try:
        print(f"Reading JSON file: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Check if JSON contains required keys
        required_keys = ['kick', 'snare', 'hihat']
        for key in required_keys:
            if key not in data:
                print(f"Error: Missing '{key}' key in JSON file")
                return None

        # Return key-value pairs
        return {
            'kick': data['kick'],
            'snare': data['snare'],
            'hihat': data['hihat']
        }

    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def update_arduino_code(arduino_path, pattern_data, backup=True):
    """
    Update hardcoded drum patterns in Arduino code
    """
    try:
        print(f"Updating Arduino code: {arduino_path}")

        # First read the entire file
        with open(arduino_path, 'r') as f:
            code = f.read()

        # Create backup if needed
        if backup:
            backup_path = f"{arduino_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(code)
            print(f"Created backup file: {backup_path}")

        # Define regex patterns for each drum pattern
        pattern_regexes = {
            'kick': r'const bool kickPatternData\[\d+\] = \{[^}]*\};',
            'snare': r'const bool snarePatternData\[\d+\] = \{[^}]*\};',
            'hihat': r'const bool hihatPatternData\[\d+\] = \{[^}]*\};'
        }

        # Create new code lines for each drum pattern
        new_pattern_lines = {
            'kick': f"const bool kickPatternData[{len(pattern_data['kick'])}] = {{{','.join(str(x) for x in pattern_data['kick'])}}};",
            'snare': f"const bool snarePatternData[{len(pattern_data['snare'])}] = {{{','.join(str(x) for x in pattern_data['snare'])}}};",
            'hihat': f"const bool hihatPatternData[{len(pattern_data['hihat'])}] = {{{','.join(str(x) for x in pattern_data['hihat'])}}};",
        }

        # Replace each pattern
        for key, regex in pattern_regexes.items():
            pattern = re.compile(regex)
            match = pattern.search(code)

            if match:
                code = code.replace(match.group(0), new_pattern_lines[key])
                print(f"Updated {key} drum pattern")
            else:
                print(f"Warning: Could not find {key} pattern definition in Arduino code")

        # Update patternLength variable
        pattern_length = len(pattern_data['kick'])
        pattern_length_regex = r'int patternLength = \d+;'
        pattern_length_replacement = f'int patternLength = {pattern_length};'

        code = re.sub(pattern_length_regex, pattern_length_replacement, code)
        print(f"Updated pattern length to {pattern_length}")

        # Write back to file
        with open(arduino_path, 'w') as f:
            f.write(code)

        print(f"Successfully updated Arduino code: {arduino_path}")
        return True

    except Exception as e:
        print(f"Error updating Arduino code: {e}")
        return False


def find_json_file(directory='.'):
    """
    Find JSON file in the specified directory
    """
    # Check if DEFAULT_JSON_FILE exists
    default_json_path = os.path.join(directory, DEFAULT_JSON_FILE)
    if os.path.exists(default_json_path):
        return default_json_path

    # Find all .json files
    json_files = []
    for file in os.listdir(directory):
        if file.endswith('.json'):
            json_files.append(os.path.join(directory, file))

    if not json_files:
        return None

    # If only one JSON file, return it directly
    if len(json_files) == 1:
        return json_files[0]

    # Let user choose
    print("\nFound multiple JSON files:")
    for i, file in enumerate(json_files):
        print(f"  {i + 1}. {os.path.basename(file)}")

    choice = 0
    while choice < 1 or choice > len(json_files):
        try:
            choice = int(input(f"Select JSON file to use (1-{len(json_files)}): "))
        except ValueError:
            choice = 0

    return json_files[choice - 1]


def find_arduino_file():
    """
    Find Arduino file in the sketch_apr24a folder
    """
    # Check if folder exists
    if not os.path.exists(DEFAULT_ARDUINO_FOLDER) or not os.path.isdir(DEFAULT_ARDUINO_FOLDER):
        print(f"Warning: Could not find {DEFAULT_ARDUINO_FOLDER} folder, trying current directory...")
        arduino_folder = '.'
    else:
        arduino_folder = DEFAULT_ARDUINO_FOLDER

    # Check if DEFAULT_ARDUINO_FILE exists in the folder
    default_arduino_path = os.path.join(arduino_folder, DEFAULT_ARDUINO_FILE)
    if os.path.exists(default_arduino_path):
        return default_arduino_path

    # Find all .ino files
    ino_files = []
    for file in os.listdir(arduino_folder):
        if file.endswith('.ino'):
            ino_files.append(os.path.join(arduino_folder, file))

    if not ino_files:
        return None

    # If only one .ino file, return it directly
    if len(ino_files) == 1:
        return ino_files[0]

    # Let user choose
    print("\nFound multiple Arduino files:")
    for i, file in enumerate(ino_files):
        print(f"  {i + 1}. {os.path.basename(file)}")

    choice = 0
    while choice < 1 or choice > len(ino_files):
        try:
            choice = int(input(f"Select Arduino file to update (1-{len(ino_files)}): "))
        except ValueError:
            choice = 0

    return ino_files[choice - 1]


def main():
    """
    Main function
    """
    print("=== Drum Pattern Updater Tool ===")

    # Find JSON file
    json_file = find_json_file()
    if not json_file:
        print(f"Error: No JSON files found. Please place {DEFAULT_JSON_FILE} in the current directory.")
        input("Press Enter to exit...")
        return
    print(f"Found JSON file: {json_file}")

    # Find Arduino file
    arduino_file = find_arduino_file()
    if not arduino_file:
        print(f"Error: No Arduino (.ino) files found.")
        print(f"Please make sure {DEFAULT_ARDUINO_FILE} is in the {DEFAULT_ARDUINO_FOLDER} folder.")
        input("Press Enter to exit...")
        return
    print(f"Found Arduino file: {arduino_file}")

    # Read JSON file
    pattern_data = read_json_file(json_file)

    if pattern_data:
        print("\nJSON file read successfully!")
        print(f"Pattern length: {len(pattern_data['kick'])}")
        print("Pattern preview:")
        for key in pattern_data:
            preview = ','.join(str(x) for x in pattern_data[key][:8])
            if len(pattern_data[key]) > 16:
                preview += ",...";
            print(f"  {key}: [{preview}]")

        # Update Arduino code
        print("\nUpdating Arduino code...")
        if update_arduino_code(arduino_file, pattern_data, MAKE_BACKUP):
            print("\nUpdate complete! You can now upload the Arduino code to your controller.")
        else:
            print("\nUpdate failed, please check error messages.")

    print("\nOperation completed!")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
