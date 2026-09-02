import os
import sys

# Add model_engine/ itself (not the project root) to the system path, matching how
# api/main.py loads this package — pipeline.py uses flat sibling imports internally
# (e.g. "from normalizer import ..."), so model_engine/ must be on sys.path directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "model_engine"))

from pipeline import TextProcessingPipeline

# Pathnirwana lines look like: <id>|<sinhala_text>|<romanized_transliteration>
#   e.g. sin_01_00001|කුඹුර ගොවියාට වී ලබා ගැනීමට උපකාරී වීම් වශයෙන් පිහිට වන්නකි.|kumbura goviyāṭa vī labā gænīmaṭa upakārī vīm vaśayen pihiṭa vannaki.
# The romanized transliteration column isn't used since IPA is generated from the Sinhala
# text directly via the existing G2P pipeline.
#
# The utterance ID follows a "<lang>_<speaker>_<utterance>" convention (matching common
# crowdsourced TTS corpus naming, e.g. "sin_01_00001"). The speaker segment is extracted
# as the speaker_id for multi-speaker training. Adjust SPEAKER_ID_SEGMENT if the actual
# IDs turn out to be structured differently.
SPEAKER_ID_SEGMENT = 1  # "sin_01_00001".split("_")[1] == "01"


def _extract_speaker_id(utt_id: str) -> str:
    parts = utt_id.split("_")
    if len(parts) > SPEAKER_ID_SEGMENT:
        return parts[SPEAKER_ID_SEGMENT]
    return "pathnirwana_speaker"  # fallback if an ID doesn't follow the expected pattern


def prepare_pathnirwana_dataset(input_path=None):
    print("Loading NLP Pipeline...")
    # Initialize the text-to-IPA processing engine
    pipeline = TextProcessingPipeline()

    # Define file paths for the Pathnirwana dataset. Defaults to the project's data/
    # folder (used in Colab/Drive runs), but an explicit input_path (e.g. a local test
    # file) can be passed in instead — see the command-line usage note below.
    txt_path = input_path or os.path.join(project_root, "data", "pathnirwana_dataset.txt")
    output_txt = os.path.join(project_root, "data", "pathnirwana_metadata.txt")

    # Check if the dataset index file exists before proceeding
    if not os.path.exists(txt_path):
        print(f"Error: Could not find '{txt_path}'. Please place the Pathnirwana index file in the 'data' folder.")
        return

    processed_count = 0
    error_count = 0

    print("Processing Pathnirwana transcripts to IPA. This may take a few minutes...")

    with open(txt_path, 'r', encoding='utf-8') as infile, \
         open(output_txt, 'w', encoding='utf-8') as outfile:

        for line_num, raw_line in enumerate(infile, start=1):
            utt_id = None
            try:
                line = raw_line.strip()
                if not line:
                    continue

                # Format: <id>|<sinhala_text>|<romanized_transliteration>
                parts = line.split("|")
                if len(parts) < 2:
                    continue  # skip malformed lines

                utt_id = parts[0].strip()
                transcript = parts[1].strip()

                if not utt_id or not transcript:
                    continue

                speaker_id = _extract_speaker_id(utt_id)

                # Audio files are assumed to be named after the utterance ID
                file_name = f"{utt_id}.wav"

                # Process the transcript through the NLP pipeline to get IPA
                result = pipeline.process(transcript)
                ipa_seq = result['ipa_sequence']

                # Format: wavs/filename.wav|ipa_sequence|speaker_id
                # (speaker_id is required since Pathnirwana is treated as multi-speaker)
                out_line = f"wavs/{file_name}|{ipa_seq}|{speaker_id}\n"
                outfile.write(out_line)

                processed_count += 1

                # Print progress in the terminal
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} files...")

            except Exception as e:
                print(f"Error processing line {line_num} ('{utt_id}'): {e}")
                error_count += 1

    # Final execution summary
    print("\n--- Pathnirwana Data Preprocessing Complete ---")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors encountered: {error_count} files")
    print(f"VITS metadata file saved successfully at: {output_txt}")


if __name__ == "__main__":
    # Usage:
    #   python prepare_pathnirwana_dataset.py                     -> reads data/pathnirwana_dataset.txt
    #   python prepare_pathnirwana_dataset.py /path/to/local/file -> reads that file instead (e.g. for local testing)
    cli_input_path = sys.argv[1] if len(sys.argv) > 1 else None
    prepare_pathnirwana_dataset(cli_input_path)
