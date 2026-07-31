# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json

import os
from pathlib import Path

from code.common.mlcommons.runner import ScopedQSL
from code.fields import harness as harness_fields
from code.llmlib.utils import prefix_logger as logging


from nvmitten.configurator import autoconfigure, bind
from .whisper_utils import load_audio_wav_format
import torch


def load_audio_tensor(fpath: os.PathLike, pin_memory: bool = True):
    assert os.path.exists(fpath), f"Cannot load tensor from {fpath} because it does not exist"

    ret_val, duration = load_audio_wav_format(fpath)
    ret_val = ret_val if not pin_memory else torch.tensor(ret_val).pin_memory()
    logging.debug("Loaded tensor to %s memory: %s (dtype=%s, shape=%s)",
                  "pinned" if pin_memory else "non-pinned",
                  str(fpath),
                  str(ret_val.dtype),
                  str(ret_val.shape))
    return ret_val, duration


@autoconfigure
@bind(harness_fields.tensor_path, "dataset_path")
class WhisperDataLoader(ScopedQSL):

    def pre_process(self, manifest_data):

        self.total_duration = 0
        count = 0
        self.n_mels = 128
        self.sample_rate = 16000
        self.audio = []
        self.reference = []
        self.duration = []
        self.wav_idx = []
        self.total_duration = 0
        self.input_ids_padded = []
        self.input_lengths = []
        for data in manifest_data:

            fname = data['files'][0]['fname']
            name = Path(fname).stem

            audio_fullpath_name = self.dataset_path / fname
            wave, _ = load_audio_tensor(str(audio_fullpath_name), pin_memory=True)
            logging.debug(f"key name: {name}")
            self.wav_idx.append(name)
            self.audio.append(wave)
            self.reference.append(data['transcript'])
            self.duration.append(data['files'][0]['duration'])

            self.total_duration += data['files'][0]['duration']
            assert wave.is_pinned()

        logging.debug(f"self.audio size: {len(self.audio)} total duration: {self.total_duration}")

    def __init__(self,
                 *args,
                 dataset_path: os.PathLike = Path.cwd(),
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.dataset_path = Path(dataset_path)
        self.manifest_filepath = self.dataset_path / "dev-all-repack.json"

        assert self.manifest_filepath.exists(), f"Manifest file do not exist {self.dataset_path}/dev-all.json does not exist"

        with open(self.manifest_filepath, 'r') as file:
            manifest_data = json.load(file)

        self.pre_process(manifest_data)
        assert self.total_sample_count == len(self.audio)

        logging.debug(f"self.total_sample_count: {self.total_sample_count} len(self.audio): {len(self.audio)} self.performance_sample_count: {self.performance_sample_count}")

    def create_batch(self, batch_size):
        # Yield batches
        for i in range(0, len(self), batch_size):
            yield self.audio[i:i + batch_size], self.duration[i:i + batch_size], self.reference[i:i + batch_size], self.wav_idx[i:i + batch_size]

    def __len__(self) -> int:
        """Get size of Dataset"""
        return len(self.audio)

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass
