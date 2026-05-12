import os
import sys
import numpy as np
import re
import toml

config_file = "config.toml"
config = toml.load(config_file)

npz_dir = config["coverage_splitting"]["npz_dir"]
export_dir = config["coverage_splitting"]["export_dir"]
threshold = config["coverage_splitting"]["threshold"]
rounding_decimals = config["coverage_splitting"]["rounding_decimals"]

# For each block, get average coverage and save identical coverages to new .npz file
def split_file(filepath):
    file = np.load(filepath, allow_pickle=True)
    # Get all blocks
    blocks = np.where(np.diff(file["timestamps"]) > threshold)[0]
    print(f"Found {len(blocks)+1} blocks in {filepath}.")  # +1 (0 is also a block start)
    
    grouped_patterns = {}
    grouped_timestamps = {}
    
    # Get average coverage in each block
    for i in range(len(blocks) + 1):
        if i == 0:
            start = 0
            end = blocks[i]
        elif i == len(blocks):
            start = blocks[i - 1]
            end = len(file["patterns"])
        else:
            start = blocks[i - 1]
            end = blocks[i]

        block_patterns = file["patterns"][start:end]
        block_timestamps = file["timestamps"][start:end]

        # Unpack all patterns
        unpacked = np.array([np.unpackbits(np.array(pattern, dtype=np.uint8), bitorder="big") for pattern in block_patterns])
        # and average coverage
        avg_coverage = np.round(np.mean(unpacked), decimals = rounding_decimals)
        coverage_str = f"{int(avg_coverage * 100):03d}"
        print(f"Block {i}: {avg_coverage} -> {coverage_str}")

        # Group patterns and timestamps by coverage
        if coverage_str not in grouped_patterns:
            grouped_patterns[coverage_str] = []
            grouped_timestamps[coverage_str] = []

        grouped_patterns[coverage_str].append(block_patterns)
        grouped_timestamps[coverage_str].append(block_timestamps)

    base_filename = os.path.basename(filepath)

    # Save each coverage group to a new .npz file
    for coverage_str in grouped_patterns:
        patterns = np.concatenate(grouped_patterns[coverage_str])
        timestamps = np.concatenate(grouped_timestamps[coverage_str])

        # Replace poisXXX or poisXXX-YYY with split_poisXXX
        new_filename = re.sub(r"pois\d{3}(?:-\d{3})?", f"split_pois{coverage_str}", base_filename)
        output_path = os.path.join(export_dir, new_filename)
        np.savez(output_path, patterns=patterns, timestamps=timestamps)

        print(f"Saved: {output_path}")

if __name__ == "__main__":
    os.makedirs(export_dir, exist_ok=True)
    for filename in os.listdir(npz_dir):
        if filename.endswith(".npz"):
            filepath = os.path.join(npz_dir, filename)
            print(f"\nProcessing {filepath}...")
            split_file(filepath)