"""
Trains a multi-speaker VITS model on the Pathnirwana dataset.

Expects, relative to the project root:
  data/pathnirwana_metadata.txt  -> produced by data_preparation/prepare_pathnirwana_dataset.py
  data/wavs/*.wav                 -> the audio referenced by pathnirwana_metadata.txt

Usage (from the project root, e.g. in Colab after `%cd` into it):
  python model_training/train_pathnirwana.py
"""

import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from formatters import pathnirwana_formatter
from vocab_utils import build_characters_config


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data")
    output_path = os.path.join(current_dir, "runs", "pathnirwana")

    METADATA_FILE = "pathnirwana_metadata.txt"

    dataset_config = BaseDatasetConfig(
        formatter="pathnirwana_formatter",
        dataset_name="voicelk_pathnirwana",
        meta_file_train=METADATA_FILE,
        path=data_path,
    )

    characters_config = build_characters_config(os.path.join(data_path, METADATA_FILE))

    # NOTE: sample_rate must match your actual .wav files. Check with, e.g.:
    #   python -c "import soundfile as sf; print(sf.info('data/wavs/<some_file>.wav'))"
    # and adjust here if it isn't 22050.
    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    vitsArgs = VitsArgs(use_speaker_embedding=True)

    config = VitsConfig(
        model_args=vitsArgs,
        audio=audio_config,
        run_name="voicelk_vits_pathnirwana",
        batch_size=16,
        eval_batch_size=8,
        batch_group_size=5,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        characters=characters_config,
        cudnn_benchmark=False,
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        formatter=pathnirwana_formatter,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.model_args.num_speakers = speaker_manager.num_speakers

    model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
