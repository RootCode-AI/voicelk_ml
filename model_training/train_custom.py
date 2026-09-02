"""
Trains a single-speaker VITS model on VoiceLK's custom (YouTube-scraped) dataset.

Expects, relative to the project root:
  data/metadata.txt   -> produced by data_preparation/prepare_dataset.py
  data/wavs/*.wav      -> the audio referenced by metadata.txt

Usage (from the project root, e.g. in Colab after `%cd` into it):
  python model_training/train_custom.py
"""

import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from formatters import custom_formatter
from vocab_utils import build_characters_config

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "data")
output_path = os.path.join(current_dir, "runs", "custom")

METADATA_FILE = "metadata.txt"

dataset_config = BaseDatasetConfig(
    formatter="custom_formatter",
    dataset_name="voicelk_custom",
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

config = VitsConfig(
    audio=audio_config,
    run_name="voicelk_vits_custom",
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
    cudnn_benchmark=True,
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    formatter=custom_formatter,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = Vits(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
