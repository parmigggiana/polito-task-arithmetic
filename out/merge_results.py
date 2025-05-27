import json
import glob
import os

def merge_json_results():
    """
    Merges all JSON files in the current directory matching the pattern
    '[DATASET_NAME]_[finetuned|base]_results.json' into a single
    'results.json' file.
    """
    merged_data = {}

    # Pattern to match the files
    # e.g., SST-2_finetuned_results.json, MNLI_base_results.json
    file_pattern = "*_results.json"

    for filepath in glob.glob(file_pattern):
        filename = os.path.basename(filepath)

        # Ensure we don't process the output file if it already exists and matches the pattern
        if filename == "results.json":
            continue

        # Extract the key for the merged dictionary
        # e.g., "SST-2_finetuned" from "SST-2_finetuned_results.json"
        if filename.endswith("_results.json"):
            key_name = filename[:-len("_results.json")] # Remove "_results.json"
        else:
            # Skip files not matching the expected suffix, though glob should handle this
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            merged_data[key_name] = data
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}. Skipping.")
        except Exception as e:
            print(f"Warning: An error occurred while processing {filename}: {e}. Skipping.")

    merged_data = sorted(merged_data.items(), key=lambda x: x[0])  # Sort by key name
    merged_data = {k: v for k, v in merged_data}  # Convert back to dict

    output_filename = "results.json"
    try:
        with open(output_filename, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)
        print(f"Successfully merged results into {output_filename}")
        for filepath in glob.glob(file_pattern):
            if filepath != output_filename:
                os.remove(filepath)
    except IOError:
        print(f"Error: Could not write to {output_filename}")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")

if __name__ == '__main__':
    merge_json_results()