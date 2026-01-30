import os
import librosa
from pathlib import Path
import sys
from collections import defaultdict, Counter

def get_audio_info(file_path):
    """
    Get comprehensive audio file information including duration, sample rate, channels, and format.

    Args:
        file_path (str): Path to the audio file

    Returns:
        dict: Dictionary containing audio information, or None if file cannot be processed
    """
    try:
        # Get duration without loading the full audio
        duration = librosa.get_duration(path=file_path)

        # Get sample rate and load a small portion to get channel info
        y, sr = librosa.load(file_path, sr=None, duration=0.1)  # Load only 0.1 seconds

        # Determine number of channels
        if y.ndim == 1:
            channels = 1
        else:
            channels = y.shape[0] if y.shape[0] < y.shape[1] else y.shape[1]

        # Get file format from extension
        file_format = Path(file_path).suffix.lower().lstrip('.')

        return {
            'duration': duration,
            'sample_rate': sr,
            'channels': channels,
            'format': file_format
        }
    except Exception as e:
        print(f"Warning: Could not process {file_path}: {e}")
        return None

def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path (str): Path to the audio file

    Returns:
        float: Duration in seconds, or 0 if file cannot be processed
    """
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Warning: Could not process {file_path}: {e}")
        return 0

def format_duration(seconds):
    """
    Format duration from seconds to human-readable format.

    Args:
        seconds (float): Duration in seconds

    Returns:
        str: Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes:02d}:{secs:06.3f}"

def is_audio_file(file_path):
    """
    Check if a file is an audio file based on its extension.

    Args:
        file_path (str): Path to the file

    Returns:
        bool: True if it's an audio file, False otherwise
    """
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus'}
    return Path(file_path).suffix.lower() in audio_extensions

def calculate_directory_stats(root_directory):
    """
    Calculate audio file statistics for each subdirectory.

    Args:
        root_directory (str): Path to the root directory to analyze

    Returns:
        dict: Dictionary with directory names as keys and stats as values
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        print(f"Error: Directory '{root_directory}' does not exist.")
        return {}

    directory_stats = defaultdict(lambda: {
        'total_duration': 0,
        'file_count': 0,
        'files': [],
        'formats': Counter(),
        'sample_rates': Counter(),
        'channels': Counter()
    })

    # Walk through all directories and subdirectories
    for current_dir, subdirs, files in os.walk(root_path):
        current_dir_path = Path(current_dir)

        # Get relative path from root directory
        if current_dir_path == root_path:
            dir_name = "ROOT"
        else:
            dir_name = current_dir_path.relative_to(root_path).as_posix()

        audio_files = [f for f in files if is_audio_file(f)]

        if audio_files:  # Only process directories with audio files
            print(f"Processing directory: {dir_name}")

            for audio_file in audio_files:
                file_path = current_dir_path / audio_file
                audio_info = get_audio_info(str(file_path))

                if audio_info:
                    duration = audio_info['duration']
                    sample_rate = audio_info['sample_rate']
                    channels = audio_info['channels']
                    file_format = audio_info['format']

                    directory_stats[dir_name]['total_duration'] += duration
                    directory_stats[dir_name]['file_count'] += 1
                    directory_stats[dir_name]['files'].append({
                        'name': audio_file,
                        'duration': duration,
                        'sample_rate': sample_rate,
                        'channels': channels,
                        'format': file_format
                    })

                    # Update counters
                    directory_stats[dir_name]['formats'][file_format] += 1
                    directory_stats[dir_name]['sample_rates'][sample_rate] += 1
                    directory_stats[dir_name]['channels'][channels] += 1

                    print(f"  - {audio_file}: {format_duration(duration)} | {file_format.upper()} | {sample_rate}Hz | {channels}ch")
                else:
                    # Still count the file even if we couldn't get detailed info
                    directory_stats[dir_name]['file_count'] += 1
                    print(f"  - {audio_file}: ERROR (could not process)")

    return directory_stats

def print_summary(directory_stats):
    """
    Print a comprehensive summary of the audio file statistics.

    Args:
        directory_stats (dict): Dictionary containing directory statistics
    """
    if not directory_stats:
        print("No audio files found in the directory structure.")
        return

    print("\n" + "="*100)
    print("AUDIO DATASET STATISTICS SUMMARY")
    print("="*100)

    total_files = 0
    total_duration = 0
    overall_formats = Counter()
    overall_sample_rates = Counter()
    overall_channels = Counter()

    # Sort directories by name for consistent output
    for dir_name in sorted(directory_stats.keys()):
        stats = directory_stats[dir_name]
        duration = stats['total_duration']
        file_count = stats['file_count']

        total_files += file_count
        total_duration += duration

        # Update overall counters
        for fmt, count in stats['formats'].items():
            overall_formats[fmt] += count
        for sr, count in stats['sample_rates'].items():
            overall_sample_rates[sr] += count
        for ch, count in stats['channels'].items():
            overall_channels[ch] += count

        print(f"\nDirectory: {dir_name}")
        print(f"  Total files: {file_count}")
        print(f"  Total duration: {format_duration(duration)} ({duration:.3f} seconds)")
        print(f"  Average duration: {format_duration(duration/file_count) if file_count > 0 else '0:00:00.000'}")

        # Format breakdown
        if stats['formats']:
            format_details = []
            for fmt, count in sorted(stats['formats'].items()):
                format_details.append(f"{count} files: {fmt.upper()}")
            print(f"  Formats: {', '.join(format_details)}")

        # Sample rate breakdown
        if stats['sample_rates']:
            sr_details = []
            for sr, count in sorted(stats['sample_rates'].items()):
                sr_freq = f"{int(sr/1000)}kHz" if sr >= 1000 else f"{int(sr)}Hz"
                sr_details.append(f"{count} files: {sr_freq}")
            print(f"  Sample rates: {', '.join(sr_details)}")

        # Channel breakdown
        if stats['channels']:
            ch_details = []
            for ch, count in sorted(stats['channels'].items()):
                ch_details.append(f"{count} files: {ch} channel{'s' if ch > 1 else ''}")
            print(f"  Channels: {', '.join(ch_details)}")

    print("\n" + "-"*100)
    print("OVERALL TOTALS:")
    print(f"  Total directories: {len(directory_stats)}")
    print(f"  Total audio files: {total_files}")
    print(f"  Total duration: {format_duration(total_duration)} ({total_duration:.3f} seconds)")
    print(f"  Average duration per file: {format_duration(total_duration/total_files) if total_files > 0 else '0:00:00.000'}")

    # Overall format breakdown
    if overall_formats:
        format_details = []
        for fmt, count in sorted(overall_formats.items()):
            format_details.append(f"{count} files: {fmt.upper()}")
        print(f"  Overall formats: {', '.join(format_details)}")

    # Overall sample rate breakdown
    if overall_sample_rates:
        sr_details = []
        for sr, count in sorted(overall_sample_rates.items()):
            sr_freq = f"{int(sr/1000)}kHz" if sr >= 1000 else f"{int(sr)}Hz"
            sr_details.append(f"{count} files: {sr_freq}")
        print(f"  Overall sample rates: {', '.join(sr_details)}")

    # Overall channel breakdown
    if overall_channels:
        ch_details = []
        for ch, count in sorted(overall_channels.items()):
            ch_details.append(f"{count} files: {ch} channel{'s' if ch > 1 else ''}")
        print(f"  Overall channels: {', '.join(ch_details)}")

    print("="*100)

def main():
    """
    Main function to run the audio statistics calculation.
    """
    # Use current directory if no argument provided
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = "."

    print(f"Analyzing audio files in: {os.path.abspath(target_directory)}")
    print("-" * 80)

    # Calculate statistics
    stats = calculate_directory_stats(target_directory)

    # Print summary
    print_summary(stats)

if __name__ == "__main__":
    main()
