# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from functools import lru_cache
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Iterable, Optional, Tuple, Union
import subprocess as sp


import numpy as np
import soundfile
import torch
import torch.nn.functional as F
import re
from num2words import num2words


Pathlike = Union[str, Path]

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

# TODO v6.0 check if identical
# Same as the 'labels' from ./3rdparty/mlc-inference/speech2text/accuracy_eval.py
labels = [
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'"]
labels_dict = {}
for i in range(len(labels)):
    labels_dict[labels[i]] = i


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac",
        "1", "-acodec", "pcm_s16le", "-ar",
        str(sr), "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio_wav_format(wav_path):
    # make sure audio in .wav format
    # assert wav_path.endswith(
    #        '.wav'), f"Only support .wav format, but got {wav_path}"
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    return waveform, sample_rate


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis,
                                       index=torch.arange(length,
                                                          device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array,
                          [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device,
                n_mels: int,
                mel_filters_dir: str = None) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    if mel_filters_dir is None:
        mel_filters_path = os.path.join(os.path.dirname(__file__), "assets",
                                        "mel_filters.npz")
    else:
        mel_filters_path = os.path.join(mel_filters_dir, "mel_filters.npz")
    with np.load(mel_filters_path) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(torch.float64).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    return_duration: bool = False,
    mel_filters_dir: str = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80 or 128, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            if audio.endswith('.wav'):
                audio, _ = load_audio_wav_format(audio)
            else:
                audio = load_audio(audio)
        assert isinstance(audio,
                          np.ndarray), f"Unsupported audio type: {type(audio)}"
        duration = audio.shape[-1] / SAMPLE_RATE
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        # pad to N_SAMPLES
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio,
                      N_FFT,
                      HOP_LENGTH,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = mel_filters(audio.device, n_mels, mel_filters_dir)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if return_duration:
        return log_spec, duration
    else:
        return log_spec


def store_transcripts(filename: Pathlike, texts: Iterable[Tuple[str, str,
                                                                str]]) -> None:
    """Save predicted results and reference transcripts to a file.
    https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


# map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,

    bool: torch.bool,
    np.bool_: torch.bool
}


class PipelineConfig:
    STEPS = 20


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for _, x in enumerate(memory_free_info)]
    return memory_free_values


def calculate_max_engine_device_memory(engine_dict):
    max_device_memory = 0
    for engine in engine_dict.values():
        max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
    return max_device_memory


def numbers_to_text(input_string):
    """
    Finds numbers (including those with commas) in a string and converts them
    to their text format.

    Args:
        input_string (str): The string to process.

    Returns:
        str: A new string with numbers converted to text.
    """
    # Regular expression to find sequences of digits, optionally separated by commas.
    number_pattern = re.compile(r'\d{1,3}(?:,\d{3})*')

    def replace_with_text(match):
        number_str_with_commas = match.group(0)
        number_str_clean = number_str_with_commas.replace(',', '')

        try:
            number = int(number_str_clean)
            return num2words(number)
        except (ValueError, Exception) as e:
            print(f"Warning: Could not convert '{number_str_with_commas}' to words. Error: {e}")
            return number_str_with_commas

    return number_pattern.sub(replace_with_text, input_string)


def currency_to_text(input_string):
    """
    Finds currency amounts (e.g., $99.99) and converts them to their
    text format, placing "dollars" and "cents" in the correct locations.

    Args:
        input_string (str): The string to process.

    Returns:
        str: A new string with currency amounts converted to text.
    """
    # Regex to find a dollar sign, optional whitespace, and a number with commas and decimals.
    currency_pattern = re.compile(r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?')

    def replace_currency(match):
        # Get the matched string (e.g., "$1,234.56")
        currency_str = match.group(0)

        # Remove '$' and ',' to get a clean number string (e.g., "1234.56")
        clean_number_str = currency_str.strip(' $').replace(',', '')

        # Check if the number has a decimal point
        if '.' in clean_number_str:
            dollars_str, cents_str = clean_number_str.split('.')

            # Convert dollars part to words
            dollars_int = int(dollars_str)
            words_dollars = num2words(dollars_int)

            # Convert cents part to words
            cents_int = int(cents_str)
            words_cents = num2words(cents_int)

            return f"{words_dollars} dollars and {words_cents} cents"
        else:
            # If no decimal, it's just a whole dollar amount
            dollars_int = int(clean_number_str)
            words_dollars = num2words(dollars_int)

            return f"{words_dollars} dollars"

    return currency_pattern.sub(replace_currency, input_string)
