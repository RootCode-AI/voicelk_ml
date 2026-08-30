import csv
import os
import sys

# Add the project root (voicelk_ml/) to the system path so Python can find the
# model_engine package regardless of the current working directory this is run from.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model_engine.pipeline import TextProcessingPipeline

# The raw OpenSLR index file has 3 columns in this order: FileID, anonymized UserID, transcription.
# Set this to False if the actual file has no header row.
HAS_HEADER = True


def prepare_openslr_dataset():
    print("Loading NLP Pipeline...")
    # Initialize the text-to-IPA processing engine
    pipeline = TextProcessingPipeline()

    # Define file paths for the OpenSLR dataset (resolved relative to the project root,
    # not the current working directory)
    csv_path = os.path.join(project_root, "data", "openslr_dataset.csv")
    output_txt = os.path.join(project_root, "data", "openslr_metadata.txt")

    # Check if the dataset index file exists before proceeding
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'. Please place the OpenSLR index file in the 'data' folder.")
        return

    processed_count = 0
    error_count = 0

    print("Processing OpenSLR transcripts to IPA. This may take a few minutes...")

    with open(csv_path, 'r', encoding='utf-8') as csvfile, \
         open(output_txt, 'w', encoding='utf-8') as outfile:

        # OpenSLR index files are sometimes comma-separated and sometimes tab-separated,
        # so detect the delimiter instead of assuming one.
        sample = csvfile.read(4096)
        csvfile.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel  # fall back to comma-separated

        reader = csv.reader(csvfile, dialect)

        if HAS_HEADER:
            next(reader, None)  # skip the header row

        for row in reader:
            file_id = None
            try:
                # Skip malformed or empty lines
                if len(row) < 3:
                    continue

                file_id = row[0].strip()
                user_id = row[1].strip()
                transcript = row[2].strip()

                if not file_id or not transcript:
                    continue

                # OpenSLR file IDs may or may not already include the audio extension
                if not file_id.lower().endswith((".wav", ".flac", ".mp3")):
                    file_id = f"{file_id}.wav"

                # Process the transcript through the NLP pipeline to get IPA
                result = pipeline.process(transcript)
                ipa_seq = result['ipa_sequence']

                # Format: wavs/filename.wav|ipa_sequence|speaker_id
                # (speaker_id is required since OpenSLR is a multi-speaker dataset)
                line = f"wavs/{file_id}|{ipa_seq}|{user_id}\n"
                outfile.write(line)

                processed_count += 1

                # Print progress in the terminal
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} files...")

            except Exception as e:
                print(f"Error processing row '{file_id}': {e}")
                error_count += 1

    # Final execution summary
    print("\n--- OpenSLR Data Preprocessing Complete ---")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors encountered: {error_count} files")
    print(f"VITS metadata file saved successfully at: {output_txt}")


if __name__ == "__main__":
    prepare_openslr_dataset()
