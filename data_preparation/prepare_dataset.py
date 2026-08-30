import csv
import os
import sys

# Add the project root (voicelk_ml/) to the system path so Python can find the
# model_engine package regardless of the current working directory this is run from.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model_engine.pipeline import TextProcessingPipeline

def prepare_vits_dataset():
    print("Loading NLP Pipeline...")
    # Initialize the text-to-IPA processing engine
    pipeline = TextProcessingPipeline()

    # Define file paths based on the custom dataset name (resolved relative to the
    # project root, not the current working directory)
    csv_path = os.path.join(project_root, "data", "custom_dataset.csv")
    output_txt = os.path.join(project_root, "data", "metadata.txt")

    # Check if the dataset CSV exists before proceeding
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'. Please ensure the CSV file is named correctly and placed in the 'data' folder.")
        return

    processed_count = 0
    error_count = 0

    print("Processing audio transcripts to IPA. This may take a few minutes...")

    # Open the input CSV and the output metadata text file
    with open(csv_path, 'r', encoding='utf-8') as csvfile, \
         open(output_txt, 'w', encoding='utf-8') as outfile:

        # Read the CSV file as a dictionary based on headers
        reader = csv.DictReader(csvfile)

        for row in reader:
            try:
                # Extract file name and transcript (matching your Google Sheet headers)
                # NOTE: uses the code-switched column since that's what matches the actual
                # spoken audio (English brand/tech terms kept as English, not transliterated).
                file_name = row['file_name'].strip()
                transcript = row['sinhala_transcript_with_code_switch'].strip()

                # Skip empty rows to prevent pipeline errors
                if not file_name or not transcript:
                    continue

                # Process the transcript through the NLP pipeline to get IPA
                result = pipeline.process(transcript)
                ipa_seq = result['ipa_sequence']

                # Format required for VITS (LJSpeech standard): wavs/filename.wav|ipa_sequence
                line = f"wavs/{file_name}|{ipa_seq}\n"
                outfile.write(line)

                processed_count += 1

                # Print progress in the terminal
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} files...")

            except Exception as e:
                print(f"Error processing row '{file_name}': {e}")
                error_count += 1

    # Final execution summary
    print("\n--- Data Preprocessing Complete ---")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors encountered: {error_count} files")
    print(f"VITS metadata file saved successfully at: {output_txt}")

if __name__ == "__main__":
    prepare_vits_dataset()
