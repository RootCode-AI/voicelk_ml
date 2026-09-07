"""
Runs inference with a trained VITS checkpoint to synthesize speech from raw
Sinhala/English text, using VoiceLK's own normalizer + G2P pipeline (not
Coqui's built-in phonemizers) to produce the IPA text the model was trained on.

Usage:
  python model_training/synthesize.py \
      --model_dir "../Models/voicelk_vits_pathnirwana-September-02-2026_10+41AM-722d5af" \
      --text "කුඹුර ගොවියාට වී ලබා ගැනීමට උපකාරී වීම් වශයෙන් පිහිට වන්නකි." \
      --speaker_id 0 \
      --out output.wav

--model_dir must contain a config.json and a .pth checkpoint (best_model.pth,
etc.) saved together from the same training run — a checkpoint without its
matching config.json cannot be loaded reliably.
"""

import argparse
import os
import sys

import torch

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "model_engine"))

from pipeline import TextProcessingPipeline


def find_checkpoint(model_dir, explicit_name=None):
    if explicit_name:
        return os.path.join(model_dir, explicit_name)
    candidates = [
        f for f in os.listdir(model_dir) if f.endswith(".pth") and "speakers" not in f.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint (.pth) file found in '{model_dir}'.")
    # prefer best_model.pth if present, else the highest-numbered checkpoint
    if "best_model.pth" in candidates:
        return os.path.join(model_dir, "best_model.pth")
    candidates.sort()
    return os.path.join(model_dir, candidates[-1])


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech from text with a trained VoiceLK VITS model.")
    parser.add_argument("--model_dir", required=True, help="Folder containing config.json and a .pth checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint filename inside model_dir (default: auto-detect)")
    parser.add_argument("--text", required=True, help="Raw Sinhala/English text to synthesize")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker index for multi-speaker models (0-based)")
    parser.add_argument("--out", default="output.wav", help="Output .wav file path")
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"'{config_path}' not found — a checkpoint needs its matching config.json to be loaded correctly."
        )

    checkpoint_path = find_checkpoint(args.model_dir, args.checkpoint)
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    config = VitsConfig()
    config.load_json(config_path)

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # speaker_manager=None: we address speakers by raw embedding index (see --speaker_id)
    # rather than by name, since this checkpoint's speakers.pth (name->id map) isn't
    # available locally — the embedding table itself is already sized from
    # config.model_args.num_speakers regardless.
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    model.load_checkpoint(config, checkpoint_path, eval=True)
    model.eval()

    print("Loading VoiceLK NLP pipeline...")
    pipeline = TextProcessingPipeline()
    result = pipeline.process(args.text)
    ipa_text = result["ipa_sequence"]
    print(f"Normalized: {result['normalized_text']}")
    print(f"IPA:        {ipa_text}")

    token_ids = tokenizer.text_to_ids(ipa_text)
    x = torch.LongTensor(token_ids).unsqueeze(0)

    aux_input = {"x_lengths": None, "d_vectors": None, "language_ids": None, "durations": None}
    aux_input["speaker_ids"] = (
        torch.LongTensor([args.speaker_id]) if config.model_args.use_speaker_embedding else None
    )

    print("Running inference...")
    with torch.no_grad():
        outputs = model.inference(x, aux_input=aux_input)

    wav = outputs["model_outputs"][0, 0].cpu().numpy()
    ap.save_wav(wav, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
