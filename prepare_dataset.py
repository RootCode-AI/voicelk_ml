import csv
import os
import sys

# Add model_engine to the system path so Python can find internal modules like normalizer and g2p
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'model_engine'))

from model_engine.pipeline import TextProcessingPipeline

def prepare_vits_dataset():
    print("Loading NLP Pipeline...")
    # Initialize the text-to-IPA processing engine
    pipeline = TextProcessingPipeline()
    
    # Define file paths based on the custom dataset name
    csv_path = "data/custom_dataset.csv"
    output_txt = "data/metadata.txt"
    
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
                file_name = row['file_name'].strip()
                transcript = row['sinhala_transcript'].strip()
                
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