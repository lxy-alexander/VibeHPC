# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Dict, Optional
import logging
import os
import re
import shutil
import subprocess
import json
import sys
from nvmitten.configurator import bind, autoconfigure
from nvmitten.utils import run_command

from ..systems.system_list import DETECTED_SYSTEM
from ..workload import Workload
from .loadgen import submission_checker_constants, model_config
from .venv_utils import ensure_venv_ready

from .. import constants as C
from .. import paths
from ...fields import general as general_fields
from ...fields import models as model_fields
from ...fields import harness as harness_fields


G_ACC_PATTERNS = submission_checker_constants.ACC_PATTERN
G_ACC_TARGETS = model_config["accuracy-target"]
G_ACC_UPPER_LIMIT = model_config["accuracy-upper-limit"]


@dataclass
class _AccuracyScriptCommand:
    """Contains metadata for the command to invoke an MLCommons Inference accuracy script"""

    executable: str
    """str: The executable name to run. Python accuracy scripts should NOT be invoked directly (i.e. ./path/to/script.py
            via the shebang). For Python-based accuracy scripts, this value should always be "python", "python3", or
            "python3.8".
    """

    argv: List[str]
    """List[str]: List of arguments to pass to the executable. For Python scripts, this should be sys.argv."""

    env: Dict[str, str]
    """Dict[str]: Dictionary of custom environment variables to pass to the executable."""

    def __str__(self) -> str:
        argv_str = " ".join((str(elem) for elem in self.argv))
        s = f"{self.executable} {argv_str}"
        if len(self.env) > 0:
            env_str = " ".join(f"{k}={v}" for k, v in self.env.items())
            s = env_str + " " + s
        return s


@autoconfigure
@bind(model_fields.precision)
@bind(harness_fields.audit_test)
class AccuracyChecker(ABC):
    """Base class for running MLCommons Inference accuracy scripts.

    This class provides the core functionality for running accuracy checks across different MLCommons benchmarks.
    Subclasses should implement the specific command generation logic for their respective benchmarks.
    """

    def __init__(self,
                 wl: Workload,
                 mlcommons_module_path: str,
                 precision: C.Precision = C.Precision.FP32,
                 audit_test: Optional[C.AuditTest] = None):
        """Creates an AccuracyChecker

        Args:
            log_file (str): Path to the accuracy log
            benchmark_conf (Dict[str, Any]): The benchmark configuration used to generate the accuracy result
            full_benchmark_name (str): The full submission name of the benchmark
            mlcommons_module_path (str): The relative filepath of the accuracy script in the MLCommons Inference repo
        """
        if wl.audit_test01_fallback_mode:
            assert audit_test == C.AuditTest.TEST01, "audit_test01_fallback_mode can only be used with TEST01"
            self.log_file = Path("mlperf_log_accuracy_baseline.json")
        else:
            self.log_file = wl.log_dir / "mlperf_log_accuracy.json"

        self.benchmark = wl.benchmark
        self.full_benchmark_name = wl.submission_benchmark
        self.mlcommons_module_path = mlcommons_module_path
        self.precision = precision
        self.acc_metric_list = list(G_ACC_TARGETS[self.full_benchmark_name])[::2]
        self.threshold_list = list(G_ACC_TARGETS[self.full_benchmark_name])[1::2]
        self.acc_pattern_list = [G_ACC_PATTERNS[acc_metric] for acc_metric in self.acc_metric_list]

    @abstractmethod
    def get_cmd(self) -> _AccuracyScriptCommand:
        """Constructs the command to run the accuracy script

        Returns:
            _AccuracyScriptCommand: The command to run
        """
        raise NotImplementedError("Subclasses must implement this method")

    def run(self) -> List[str]:
        """Runs the accuracy checker script and returns the output if the script ran successfully.
        """
        cmd = self.get_cmd()
        if cmd.executable.startswith("python"):
            cmd.executable = self.benchmark.python_path
        return run_command(str(cmd), get_output=True)

    def get_accuracy(self) -> List[Dict[str, Any]]:
        """Runs the accuracy script and get_accuracies the accuracy results.

        Returns:
            Dict[str, Any]: A dictionary with the keys:
                - "accuracy": Float value representing the raw accuracy score
                - "threshold": Float value representing the minimum required accuracy for a valid submission
                - "pass": Bool value representing if the accuracy test passed
        """
        output = self.run()
        accuracy_result_list = []
        for i, acc_pattern in enumerate(self.acc_pattern_list):
            result_regex = re.compile(acc_pattern)
            threshold = self.threshold_list[i]

            # Copy the output to accuracy.txt
            accuracy = None
            with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w", encoding="utf-8") as f:
                for line in output:
                    print(line, file=f)

            # Extract the accuracy metric from the output
            for line in output:
                result_match = result_regex.search(line)
                if not result_match is None:
                    accuracy = float(result_match.group(1))
                    break

            passed = accuracy is not None and accuracy >= threshold
            accuracy_result_list.append({"name": self.acc_metric_list[i], "value": accuracy, "threshold": threshold, "pass": passed})
        return accuracy_result_list


@autoconfigure
@bind(general_fields.preprocessed_data_dir)
class RetinanetAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for Retinanet benchmark."""

    def __init__(self,
                 wl: Workload,
                 preprocessed_data_dir: Path = paths.BUILD_DIR / "preprocessed_data"):
        super().__init__(wl, "vision/classification_and_detection/tools/accuracy-openimages.py")
        self.openimages_dir = Path(preprocessed_data_dir) / "open-images-v6-mlperf"

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--mlperf-accuracy-file {self.log_file}",
                f"--openimages-dir {self.openimages_dir}",
                "--output-file build/retinanet-results.json"]
        return _AccuracyScriptCommand("python3", argv, dict())


@autoconfigure
@bind(general_fields.data_dir)
class BERTAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for BERT benchmark."""

    dtype_expand_map = {"fp16": "float16", "fp32": "float32", "int8": "float16"}  # Use FP16 output for INT8 mode
    """Dict[str, str]: Remap MLPINF precision strings to a string that the BERT accuracy script understands"""

    def __init__(self,
                 wl: Workload,
                 data_dir: Path = paths.BUILD_DIR / "data"):
        super().__init__(wl, "language/bert/accuracy-squad.py")
        self.squad_path = data_dir / "squad" / "dev-v1.1.json"
        self.vocab_file_path = paths.MODEL_DIR / "bert/vocab.txt"
        self.output_prediction_path = self.log_file.parent / "predictions.json"

        _dtype = self.precision.valstr.lower()
        self.dtype = BERTAccuracyChecker.dtype_expand_map[_dtype]

    def get_cmd(self) -> _AccuracyScriptCommand:
        # Having issue installing tokenizers on SoC systems. Use custom BERT accuracy script.
        if "is_soc" in DETECTED_SYSTEM.extras["tags"]:
            argv = ["code/bert/tensorrt/accuracy-bert.py",
                    f"--mlperf-accuracy-file {self.log_file}",
                    f"--squad-val-file {self.squad_path}"]
            env = dict()
        else:
            argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                    f"--log_file {self.log_file}",
                    f"--vocab_file {self.vocab_file_path}",
                    f"--val_data {self.squad_path}",
                    f"--out_file {self.output_prediction_path}",
                    f"--output_dtype {self.dtype}"]
            env = {"PYTHONPATH": "code/bert/tensorrt/helpers"}
        return _AccuracyScriptCommand("python3", argv, env)


def validate_hf_checkpoint(checkpoint_dir: str):
    """Check if the checkpoint directory is a valid Hugging Face checkpoint.
    Raise an error if the checkpoint is not valid.
    """
    required_files = ["config.json", "tokenizer.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(checkpoint_dir, file)):
            raise FileNotFoundError(f"Missing Checkpoint in: {checkpoint_dir}. Please download or move the checkpoint to the directory.")


@autoconfigure
@bind(general_fields.data_dir)
class GPTJAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for GPT-J benchmark."""

    def __init__(self,
                 wl: Workload,
                 data_dir: Path = paths.BUILD_DIR / "data"):
        super().__init__(wl, "language/gpt-j/evaluation.py")
        self.cnn_daily_mail_path = data_dir / "cnn-daily-mail" / "cnn_eval.json"

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--mlperf-accuracy-file {self.log_file}",
                f"--dataset-file {self.cnn_daily_mail_path}",
                f"--dtype int32"]
        env = dict()
        return _AccuracyScriptCommand("python3", argv, env)


@autoconfigure
@bind(general_fields.preprocessed_data_dir)
class Llama2AccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for Llama2 benchmark."""

    def __init__(self,
                 wl: Workload,
                 preprocessed_data_dir: Path = paths.BUILD_DIR / "preprocessed_data"):
        super().__init__(wl, "language/llama2-70b/evaluate-accuracy.py")

        # Check if the local model is available for faster loading.
        self.upper_limit_list = list(G_ACC_UPPER_LIMIT[self.full_benchmark_name])[1::2]
        self.ref_acc_pkl_path = preprocessed_data_dir / "open_orca" / "open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
        self.llama2_70b_ckpt_dir = paths.MODEL_DIR / "Llama2" / "Llama-2-70b-chat-hf"

        local_model_path = Path("/raid/data/mlperf-llm/Llama-2-70b-chat-hf")
        if local_model_path.exists():
            logging.info("using local Llama2 model from %s", local_model_path)
            self.llama2_70b_ckpt_dir = str(local_model_path)
        validate_hf_checkpoint(self.llama2_70b_ckpt_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--checkpoint-path {self.llama2_70b_ckpt_dir}",
                f"--mlperf-accuracy-file {self.log_file}",
                f"--dataset-file {self.ref_acc_pkl_path}",
                f"--dtype int32"]
        env = dict()
        return _AccuracyScriptCommand("python3", argv, env)


@autoconfigure
@bind(general_fields.data_dir)
class Llama3_1_8BAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for Llama3.1 benchmark."""

    def __init__(self,
                 wl: Workload,
                 data_dir: Path = paths.BUILD_DIR / "data"):
        super().__init__(wl, "language/llama3.1-8b/evaluation.py")
        self.dataset_path = data_dir / "llama3.1-8b" / "cnn_eval.json"
        self.checkpoint_dir = paths.MODEL_DIR / "Llama3.1-8B" / "Meta-Llama-3.1-8B-Instruct"

        if (local_model_path := Path("/raid/data/mlperf/llm-large/Meta-Llama-3.1-8B-Instruct")).exists():
            logging.info("using local Llama3.1 model from %s", local_model_path)
            self.checkpoint_dir = str(local_model_path)
        validate_hf_checkpoint(self.checkpoint_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--mlperf-accuracy-file {self.log_file}",
                f"--dataset-file {self.dataset_path}",
                f"--model-name {self.checkpoint_dir}",
                f"--dtype int32"]
        env = dict()
        return _AccuracyScriptCommand("python3", argv, env)


@autoconfigure
@bind(general_fields.preprocessed_data_dir)
class Llama3_1_405BAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for Llama3.1 benchmark."""

    def __init__(self,
                 wl: Workload,
                 preprocessed_data_dir: Path = paths.BUILD_DIR / "preprocessed_data"):
        super().__init__(wl, "language/llama3.1-405b/evaluate-accuracy.py")

        self.upper_limit_list = list(G_ACC_UPPER_LIMIT[self.full_benchmark_name])[1::2]
        self.dataset_path = preprocessed_data_dir / "llama3.1-405b" / "mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl"
        self.checkpoint_dir = paths.MODEL_DIR / "Llama3.1-405B" / "Meta-Llama-3.1-405B-Instruct"

        if (local_model_path := Path("/raid/data/mlperf/llm-large/Meta-Llama-3.1-405B-Instruct")).exists():
            logging.info("using local Llama3.1 model from %s", local_model_path)
            self.checkpoint_dir = str(local_model_path)
        validate_hf_checkpoint(self.checkpoint_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--checkpoint-path {self.checkpoint_dir}",
                f"--mlperf-accuracy-file {self.log_file}",
                f"--dataset-file {self.dataset_path}",
                f"--dtype int32"]
        env = dict()
        return _AccuracyScriptCommand("python3", argv, env)


@autoconfigure
@bind(general_fields.preprocessed_data_dir)
class Mixtral8x7bAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for Mixtral8x7b benchmark."""

    def __init__(self,
                 wl: Workload,
                 preprocessed_data_dir: Path = paths.MLPERF_SCRATCH_PATH / "preprocessed_data"):
        super().__init__(wl, "language/mixtral-8x7b/evaluate-accuracy.py")
        self.ref_acc_pkl_path = preprocessed_data_dir / "moe" / "mlperf_mixtral8x7b_moe_dataset_15k.pkl"
        self.upper_limit_dict = dict(zip(G_ACC_UPPER_LIMIT[self.full_benchmark_name][0::2], G_ACC_UPPER_LIMIT[self.full_benchmark_name][1::2]))

        self.mixtral_8x7b_ckpt_dir = paths.MLPERF_SCRATCH_PATH / "models" / "Mixtral" / "Mixtral-8x7B-Instruct-v0.1"

        # Check if the local model is available for faster loading.
        local_model_path = Path("/raid/data/mlperf-llm/Mixtral-8x7B-Instruct-v0.1")
        if local_model_path.exists():
            logging.info("using local model from %s", local_model_path)
            self.mixtral_8x7b_ckpt_dir = str(local_model_path)
        validate_hf_checkpoint(self.mixtral_8x7b_ckpt_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [
            f"--module-path={paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path}",
            f"--checkpoint-path={self.mixtral_8x7b_ckpt_dir}",
            f"--mlperf-accuracy-file={self.log_file}",
            f"--dataset-file={self.ref_acc_pkl_path}",
        ]
        return _AccuracyScriptCommand(str(paths.WORKING_DIR / "code/mixtral-8x7b/tensorrt/run_accuracy.sh"), argv, dict())

    def get_accuracy(self) -> List[Dict[Any, Any]]:
        """Runs the accuracy script and get_accuracys the accuracy results for Mixtral-8x7B.
           Mixtral-8x7B needs to check both the lower bound and the upper bound of TOKENS_PER_SAMPLE.

        Returns:
            Dict[str, Any]: A dictionary with the keys:
                - "accuracy": Float value representing the raw accuracy score
                - "threshold": Float value representing the minimum required accuracy for a valid submission
                - "upper_limit": Float value representing the maximum required accuracy for a valid submission
                - "pass": Bool value representing if the accuracy test passed
        """
        output = self.run()
        accuracy_result_list = []
        for i, acc_pattern in enumerate(self.acc_pattern_list):
            result_regex = re.compile(acc_pattern)
            acc_metric = self.acc_metric_list[i]
            threshold = self.threshold_list[i]

            # Copy the output to accuracy.txt
            accuracy = None
            with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w", encoding="utf-8") as f:
                for line in output:
                    print(line, file=f)

            # Extract the accuracy metric from the output
            for line in output:
                result_match = result_regex.search(line)
                if not result_match is None:
                    accuracy = float(result_match.group(1))
                    break

            upper_limit = self.upper_limit_dict.get(acc_metric, accuracy)
            passed = accuracy is not None and threshold <= accuracy <= upper_limit
            accuracy_result_list.append({
                "name": self.acc_metric_list[i],
                "value": accuracy,
                "threshold": threshold,
                "pass": passed,
            })

            if acc_metric in self.upper_limit_dict:
                accuracy_result_list[-1]['upper_limit'] = upper_limit

        return accuracy_result_list


@autoconfigure
@bind(general_fields.data_dir)
class DeepSeek_R1AccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for DeepSeek-R1 benchmark."""

    def __init__(self,
                 wl: Workload,
                 data_dir: Path = paths.BUILD_DIR / "data"):
        super().__init__(wl, "language/deepseek-r1/eval_accuracy.py")
        self.dataset_path = data_dir / "deepseek-r1" / "mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl"

        # Need to patch the PRM800k setup.py (3rdparty/mlc-inference/language/deepseek-r1/submodules/prm800k/setup.py) to remove the "import numpy" line.
        logging.info("Patching PRM800k setup.py to remove the 'import numpy' line...")
        prm800k_setup_py = paths.MLCOMMONS_INF_REPO / "language" / "deepseek-r1" / "submodules" / "prm800k" / "setup.py"
        if not prm800k_setup_py.exists():
            raise FileNotFoundError(f"PRM800k setup.py not found at {prm800k_setup_py}, please run `make clone_loadgen` first.")

        with open(prm800k_setup_py, "r") as f:
            lines = f.readlines()
        lines = [line for line in lines if "import numpy" not in line]
        with open(prm800k_setup_py, "w") as f:
            f.writelines(lines)

        # Need to instantiate a separate venv to avoid conflicts with the main venv.
        self.venv_path = Path("/work/.dsr1-acc-venv")
        requirements_file = paths.CODE_DIR / "deepseek-r1" / "tensorrt" / "requirements.accuracy.txt"
        self.venv_path = ensure_venv_ready(self.venv_path, requirements_file)

    def get_cmd(self) -> _AccuracyScriptCommand:
        output_file = self.log_file.parent / "deepseek-r1-accuracy.pkl"
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--dataset-file {self.dataset_path}",
                f"--input-file {self.log_file}",
                f"--output-file {output_file}"]
        env = dict()
        return _AccuracyScriptCommand(str(self.venv_path / "bin" / "python3"), argv, env)


class DLRMv2AccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for DLRMv2 benchmark."""

    def __init__(self, wl: Workload):
        super().__init__(wl, "recommendation/dlrm_v2/pytorch/tools/accuracy-dlrm.py")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--mlperf-accuracy-file {self.log_file}",
                "--day-23-file /home/mlperf_inf_dlrmv2/criteo/day23/raw_data",
                "--aggregation-trace-file /home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.txt",
                "--dtype float32"]
        return _AccuracyScriptCommand("python3", argv, dict())


class RGATAccuracyChecker(AccuracyChecker):
    def __init__(self, wl: Workload):
        super().__init__(wl, "graph/R-GAT/tools/accuracy_igbh.py")

        # Set up temporary directories
        dst = Path("/home/mlperf_inf_rgat/acc_checker")

        node_file = dst / "full" / "processed" / "paper" / "node_label_2K.npy"
        if not node_file.exists():
            node_file.parent.mkdir(parents=True, exist_ok=True)
            src = Path("/home/mlperf_inf_rgat/optimized/converted/graph/full/node_label_2K.npy")
            shutil.copy(src, node_file)

        val_index = dst / "full" / "processed" / "val_idx.pt"
        if not val_index.exists():
            val_index.parent.mkdir(parents=True, exist_ok=True)
            src = Path("/home/mlperf_inf_rgat/optimized/converted/graph/full/val_idx.pt")
            shutil.copy(src, val_index)

        self.acc_file_root = dst
        self.tmp_file = "/tmp/rgat_acc_results.txt"

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--mlperf-accuracy-file {self.log_file}",
                "--dataset-size full",
                "--no-memmap",
                f"--dataset-path {self.acc_file_root}",
                f"--output-file {self.tmp_file}",
                "--dtype int64"]
        return _AccuracyScriptCommand("python3", argv, dict())

    def run(self) -> List[str]:
        super().run()

        with open(self.tmp_file, 'r') as f:
            lines = f.readlines()
        return lines

    def get_accuracy(self) -> List[Dict[str, Any]]:
        """Runs the accuracy script and get_accuracies the accuracy results.

        Returns:
            Dict[str, Any]: A dictionary with the keys:
                - "accuracy": Float value representing the raw accuracy score
                - "threshold": Float value representing the minimum required accuracy for a valid submission
                - "pass": Bool value representing if the accuracy test passed
        """
        d = super().get_accuracy()[0]
        d["value"] = d["value"] / 100
        d["pass"] = (d["value"] >= d["threshold"])
        return [d]


class SDXLAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for SDXL benchmark."""

    def __init__(self, wl: Workload):
        super().__init__(wl, "text_to_image/tools/accuracy_coco.py")
        self.upper_limit_list = list(G_ACC_UPPER_LIMIT[self.full_benchmark_name])[1::2]
        self.compliance_image_path = self.log_file.parent / "images"

    def get_cmd(self) -> _AccuracyScriptCommand:
        statistics_path = paths.MLCOMMONS_INF_REPO / "text_to_image/tools/val2014.npz"
        caption_path = paths.MLCOMMONS_INF_REPO / "text_to_image/coco2014/captions/captions_source.tsv"
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--mlperf-accuracy-file {self.log_file}",
                f"--caption-path {caption_path}",
                f"--statistics-path {statistics_path}",
                "--output-file /tmp/sdxl-accuracy.json",
                f"--compliance-images-path {self.compliance_image_path}",
                "--device gpu" if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) < 100 else "--device cpu"]

        if "is_soc" in DETECTED_SYSTEM.extras["tags"]:
            argv.append("--low_memory")

        return _AccuracyScriptCommand("python3", argv, dict())

    def get_accuracy(self) -> List[Dict[str, Any]]:
        """Runs the accuracy script and get_accuracys the accuracy results for SDXL.
           SDXL needs to check both the lower bound and the upper bound of FID and CLIP

        Returns:
            Dict[str, Any]: A dictionary with the keys:
                - "accuracy": Float value representing the raw accuracy score
                - "threshold": Float value representing the minimum required accuracy for a valid submission
                - "upper_limit": Float value representing the maximum required accuracy for a valid submission
                - "pass": Bool value representing if the accuracy test passed
        """
        output = self.run()
        accuracy_result_list = []
        for i, acc_pattern in enumerate(self.acc_pattern_list):
            result_regex = re.compile(acc_pattern)
            threshold = self.threshold_list[i]
            upper_limit = self.upper_limit_list[i]

            # Copy the output to accuracy.txt
            accuracy = None
            with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w", encoding="utf-8") as f:
                for line in output:
                    print(line, file=f)

            # Extract the accuracy metric from the output
            for line in output:
                result_match = result_regex.search(line)
                if not result_match is None:
                    accuracy = float(result_match.group(1))
                    break

            passed = accuracy is not None and accuracy >= threshold and accuracy <= upper_limit
            accuracy_result_list.append({"name": self.acc_metric_list[i], "value": accuracy, "threshold": threshold, "upper_limit": upper_limit, "pass": passed})
        return accuracy_result_list


class WhisperAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for Whisper benchmark."""

    def __init__(self, wl: Workload):
        super().__init__(wl, "speech2text/accuracy_eval.py")
        self.log_dir = wl.log_dir

        self.acc_metric_list = list(G_ACC_TARGETS[self.full_benchmark_name])[::2]
        self.acc_pattern_list = [G_ACC_PATTERNS[acc_metric] for acc_metric in self.acc_metric_list]
        self.threshold_list = list(G_ACC_TARGETS[self.full_benchmark_name])[1::2]

    def get_cmd(self):
        cmd = "python3"
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--log_dir {self.log_dir}",
                f"--dataset_dir {paths.BUILD_DIR}/preprocessed_data/whisper-large-v3/dev-all-repack/",
                f"--manifest {paths.BUILD_DIR}/preprocessed_data/whisper-large-v3/dev-all-repack.json",
                "--output_dtype int8",
                ]

        env = dict()

        return _AccuracyScriptCommand(cmd, argv, env)

    def get_accuracy(self) -> List[Dict[str, Any]]:

        try:
            wer_string = self.run()
        except Exception as e:
            logging.error(f"Accuracy run FAILED: {e}")
        with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w", encoding="utf-8") as f:
            for line in wer_string:
                print(line, file=f)
        accuracy_result_list = []
        for i, acc_pattern in enumerate(self.acc_pattern_list):
            result_regex = re.compile(acc_pattern)
            threshold = self.threshold_list[i]
            for line in wer_string:
                result_match = result_regex.search(line)
                if not result_match is None:
                    accuracy = float(result_match.group(1))
                    passed = accuracy >= threshold
                    accuracy_result_list.append({"name": self.acc_metric_list[0], "value": accuracy, "threshold": threshold, "pass": passed})
        return accuracy_result_list


class Q3VLAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for Qwen3-VL-235B-A22B benchmark."""

    def __init__(self, wl: Workload):
        super().__init__(wl, "multimodal/qwen3-vl/evaluate.py")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = ["evaluate", f"--filename {self.log_file}"]
        return _AccuracyScriptCommand("mlperf-inf-mm-q3vl", argv, dict())

    def run(self) -> List[str]:
        """Run Q3VL accuracy checker, capturing stderr for CLI output."""
        cmd = self.get_cmd()
        return run_command(f"{str(cmd)} 2>&1", get_output=True)

    def get_accuracy(self) -> List[Dict[str, Any]]:
        """Run Q3VL accuracy evaluation and parse F1_HIERARCHICAL."""
        self.run()

        # The upstream evaluation writes accuracy.txt to the current working dir.
        log_accuracy_path = Path(os.path.dirname(self.log_file)) / "accuracy.txt"
        cwd_accuracy_path = Path("accuracy.txt")
        if cwd_accuracy_path.exists() and cwd_accuracy_path.resolve() != log_accuracy_path.resolve():
            shutil.move(str(cwd_accuracy_path), str(log_accuracy_path))

        acc_metric = self.acc_metric_list[0]
        threshold = self.threshold_list[0]
        result_regex = re.compile(self.acc_pattern_list[0])

        accuracy = None
        if log_accuracy_path.exists():
            with log_accuracy_path.open("r", encoding="utf-8") as f:
                for line in f:
                    result_match = result_regex.search(line)
                    if result_match is not None:
                        accuracy = float(result_match.group(1))
                        break
                    stripped = line.strip()
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            accuracy = float(json.loads(stripped).get("f1"))
                            break
                        except (ValueError, TypeError, json.JSONDecodeError):
                            pass

            if accuracy is None:
                # Fallback parsing for common Q3VL output formats.
                fallback_regex = re.compile(
                    r"(?:F1_HIERARCHICAL|Category hierarchical F1 Score)\s*[:=]\s*([0-9]*\.?[0-9]+)",
                    re.IGNORECASE,
                )
                with log_accuracy_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        result_match = fallback_regex.search(line)
                        if result_match is not None:
                            accuracy = float(result_match.group(1))
                            break

        passed = accuracy is not None and accuracy >= threshold
        return [{
            "name": acc_metric,
            "value": accuracy,
            "threshold": threshold,
            "pass": passed,
        }]


class Wan22AccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for WAN22-A14B text-to-video benchmark.

    Uses VBench evaluation infrastructure from the MLCommons inference repository.
    """

    def __init__(self, wl: Workload):
        super().__init__(wl, "text_to_video/wan-2.2-t2v-a14b/run_evaluation.py")

        # WAN22-specific paths
        self.log_dir = wl.log_dir
        self.video_dir = self.log_dir / "video"
        self.eval_output_dir = self.log_dir / "vbench_results"

        # VBench environment setup
        self.venv_path = Path("/work/.vbench-venv")
        self.vbench_submodule_path = paths.MLCOMMONS_INF_REPO / "text_to_video/wan-2.2-t2v-a14b/submodules/VBench"
        self._machine = os.uname().machine  # same as uname -m (e.g. x86_64, aarch64)
        self._use_gb_venv = self._machine == "aarch64"
        accuracy_dir = Path(__file__).parent.parent.parent / "wan22-a14b" / "tensorrt" / "accuracy"
        self.vbench_requirements = accuracy_dir / f"vbench_requirements.{self._machine}.txt"

        self._setup_vbench_env()

    def _setup_vbench_env(self):
        """Set up the VBench virtual environment."""
        if self.venv_path.exists():
            pip_path = self.venv_path / "bin" / "pip"
            result = subprocess.run(
                [str(pip_path), "show", "vbench"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
            )
            if result.returncode != 0:
                logging.warning(f"VBench not installed in {self.venv_path}. Installing...")
                self._install_vbench_deps()
            else:
                logging.info(f"VBench environment found at {self.venv_path}")
        else:
            logging.info(f"Creating VBench virtual environment at {self.venv_path}...")
            venv_cmd = [sys.executable, "-m", "venv", str(self.venv_path)]
            if self._use_gb_venv:
                venv_cmd.insert(-1, "--system-site-packages")
            subprocess.run(venv_cmd, check=True)
            self._install_vbench_deps()

    def _install_vbench_deps(self):
        """Install VBench dependencies in the virtual environment."""
        pip_path = self.venv_path / "bin" / "pip"

        logging.info("Installing VBench dependencies...")
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(pip_path), "install", "-r", str(self.vbench_requirements)], check=True)
        subprocess.run([str(pip_path), "install", "vbench", "--no-deps"], check=True)

        # Fix OpenCV for headless environments
        subprocess.run([str(pip_path), "uninstall", "-y", "opencv-python"], check=False)
        subprocess.run([str(pip_path), "install", "--force-reinstall", "opencv-python-headless>=4.8.0"], check=True)
        logging.info("VBench environment setup complete")

    def get_cmd(self) -> _AccuracyScriptCommand:
        vbench_script = self.vbench_submodule_path / "evaluate.py"
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)

        # VBench evaluation dimensions for WAN22-A14B
        # Using standard mode (not custom_input) to support all dimensions including appearance_style and scene
        vbench_dimensions = [
            "subject_consistency",
            "dynamic_degree",
            "motion_smoothness",
            "appearance_style",
            "scene",
            "background_consistency",
        ]

        argv = [
            str(vbench_script),
            f"--videos_path={self.video_dir}",
            f"--output_path={self.eval_output_dir}",
            "--load_ckpt_from_local=True",
            "--dimension",
        ] + vbench_dimensions

        return _AccuracyScriptCommand(
            str(self.venv_path / "bin" / "python3"),
            argv,
            {
                "PYTHONPATH": str(self.vbench_submodule_path),
                "LD_LIBRARY_PATH": f"/usr/lib/{self._machine}-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", ""),
            },
        )

    def run(self) -> List[str]:
        """Run the VBench evaluation and return output lines."""
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")

        video_files = list(self.video_dir.glob("*.mp4"))
        if not video_files:
            raise FileNotFoundError(f"No video files found in {self.video_dir}")

        logging.info(f"Found {len(video_files)} videos for evaluation")

        cmd = self.get_cmd()
        logging.info(f"Running VBench evaluation: {cmd}")

        result = subprocess.run(
            [cmd.executable] + cmd.argv,
            env={**os.environ, **cmd.env},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"VBench evaluation failed: {result.stdout}")

        return self._parse_vbench_results()

    def _parse_vbench_results(self) -> List[str]:
        """Parse VBench results and return formatted output lines."""
        result_files = sorted(self.eval_output_dir.glob("results_*_eval_results.json"))
        if not result_files:
            return ["VBench evaluation completed but no results found"]

        result_file = result_files[-1]
        logging.info(f"Reading results from {result_file}")

        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        output_lines = ["=" * 60, "VBench Evaluation Results", "=" * 60, "", "Dimension Scores:", "-" * 60]

        total_score = 0.0
        num_dimensions = 0
        for dimension, value in sorted(results.items()):
            if isinstance(value, list) and len(value) > 0:
                score = value[0]
                if isinstance(score, (int, float)):
                    total_score += score
                    num_dimensions += 1
                    output_lines.append(f"  {dimension:30s}: {score:6.4f}")

        if num_dimensions > 0:
            overall_avg = total_score / num_dimensions
            # VBench scores are in [0,1], multiply by 100 for percentage comparison
            overall_avg_pct = overall_avg * 100
            threshold = self.threshold_list[0]
            # Line for submission checker: must match ACC_PATTERN["vbench_score"] (e.g. vbench_score: 70.48)
            # Line for submission checker: must match ACC_PATTERN["vbench_score"] (e.g. 'vbench_score': 70.48)
            output_lines.append(f"'vbench_score': {overall_avg_pct:.4f}")
            output_lines.extend([
                "-" * 60,
                f"  {'Overall Average':30s}: {overall_avg:6.4f} ({overall_avg_pct:.2f}%)",
                "",
                f"Threshold: {threshold:.4f}%",
                f"Pass: {'Yes' if overall_avg_pct >= threshold else 'No'}",
            ])

        output_lines.extend(["=" * 60, f"Detailed results: {result_file}", "=" * 60])

        return output_lines

    def get_accuracy(self) -> List[Dict[str, Any]]:
        """Run VBench evaluation and return accuracy results."""
        output = self.run()

        # Write output to accuracy.txt
        with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w", encoding="utf-8") as f:
            for line in output:
                print(line, file=f)

        # Extract overall score from results
        result_files = sorted(self.eval_output_dir.glob("results_*_eval_results.json"))
        overall_score = None
        if result_files:
            with open(result_files[-1], "r", encoding="utf-8") as f:
                results = json.load(f)
            total_score = 0.0
            num_dimensions = 0
            for value in results.values():
                if isinstance(value, list) and len(value) > 0:
                    score = value[0]
                    if isinstance(score, (int, float)):
                        total_score += score
                        num_dimensions += 1
            if num_dimensions > 0:
                # VBench scores are in [0,1], multiply by 100 for percentage comparison
                overall_score = (total_score / num_dimensions) * 100

        threshold = self.threshold_list[0]
        passed = overall_score is not None and overall_score >= threshold

        return [{
            "name": self.acc_metric_list[0],
            "value": overall_score,
            "threshold": threshold,
            "pass": passed,
        }]
@autoconfigure
@bind(general_fields.data_dir)
class GptOss120bAccuracyChecker(AccuracyChecker):
    """Accuracy checker implementation for GPT-OSS-120B benchmark."""

    def __init__(self,
                 wl: Workload,
                 data_dir: Path = paths.BUILD_DIR / "data"):
        super().__init__(wl, "language/gpt-oss-120b/eval_mlperf_accuracy.py")

        # Reference data file - check in data/gpt-oss/v4/acc/
        self.ref_data_path = data_dir / "gpt-oss" / "v4" / "acc" / "acc_eval_ref.parquet"

        # Tokenizer - use HuggingFace model name
        self.tokenizer_name = "openai/gpt-oss-120b"

        # Check for local model path for faster loading
        local_model_path = Path("/raid/data/mlperf-llm/gpt-oss-120b")
        if local_model_path.exists():
            logging.info("using local GPT-OSS-120B model from %s", local_model_path)
            self.tokenizer_name = str(local_model_path)

        # Check for upper limits (for TOKENS_PER_SAMPLE)
        if self.full_benchmark_name in G_ACC_UPPER_LIMIT:
            self.upper_limit_list = list(G_ACC_UPPER_LIMIT[self.full_benchmark_name])[1::2]
        else:
            self.upper_limit_list = []

        # Set up venv for accuracy checker (avoids modifying base image)
        self.venv_path = Path("/work/.gptoss-acc-venv")
        requirements_file = paths.CODE_DIR / "gpt-oss-120b" / "tensorrt" / "requirements.accuracy.txt"
        self.venv_path = ensure_venv_ready(self.venv_path, requirements_file)

    def get_cmd(self) -> _AccuracyScriptCommand:
        output_file = self.log_file.parent / "gpt-oss-120b-accuracy.json"
        argv = [paths.MLCOMMONS_INF_REPO / self.mlcommons_module_path,
                f"--mlperf-log {self.log_file}",
                f"--reference-data {self.ref_data_path}",
                f"--tokenizer {self.tokenizer_name}",
                f"--output-file {output_file}"]
        env = dict()
        return _AccuracyScriptCommand(str(self.venv_path / "bin" / "python3"), argv, env)


G_ACCURACY_CHECKER_MAP = {C.Benchmark.BERT: BERTAccuracyChecker,
                          C.Benchmark.DLRMv2: DLRMv2AccuracyChecker,
                          C.Benchmark.GPTJ: GPTJAccuracyChecker,
                          C.Benchmark.LLAMA2: Llama2AccuracyChecker,
                          C.Benchmark.LLAMA3_1_8B: Llama3_1_8BAccuracyChecker,
                          C.Benchmark.LLAMA3_1_405B: Llama3_1_405BAccuracyChecker,
                          C.Benchmark.Mixtral8x7B: Mixtral8x7bAccuracyChecker,
                          C.Benchmark.DeepSeek_R1: DeepSeek_R1AccuracyChecker,
                          C.Benchmark.GPT_OSS_120B: GptOss120bAccuracyChecker,
                          C.Benchmark.Q3VL: Q3VLAccuracyChecker,
                          C.Benchmark.Retinanet: RetinanetAccuracyChecker,
                          C.Benchmark.RGAT: RGATAccuracyChecker,
                          C.Benchmark.SDXL: SDXLAccuracyChecker,
                          C.Benchmark.WHISPER: WhisperAccuracyChecker,
                          C.Benchmark.WAN22_A14B: Wan22AccuracyChecker}
"""Dict[Benchmark, AccuracyChecker]: Maps a Benchmark to its AccuracyChecker"""


def check_accuracy(wl: Workload):
    """Check accuracy of given benchmark."""
    # Check if log_file is empty by just reading first several bytes
    # The first 4B~6B is likely all we need to check: '', '[]', '[]\r', '[\n]\n', '[\r\n]\r\n', ...
    # but checking 8B for safety
    with (wl.log_dir / "mlperf_log_accuracy.json").open(mode='r') as lf:
        first_8b = lf.read(8)
        if not first_8b or ('[' in first_8b and ']' in first_8b):
            return "No accuracy results in PerformanceOnly mode."

    checker_cls = G_ACCURACY_CHECKER_MAP[wl.benchmark]
    accuracy_checker = checker_cls(wl)
    return accuracy_checker.get_accuracy()


# Provide a way to call accuracy checker separately from commandline, so we can
# check the functionality of the accuracy checker without running the whole build process.
if __name__ == "__main__":
    import argparse
    import sys

    # Set up logging to print debug info to stdout
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run MLCommons accuracy checker. "
                    "Example for DeepSeek-R1:\n"
                    "python3 -m code.common.mlcommons.accuracy_checker "
                    "--benchmark DeepSeek_R1 --scenario Offline --precision FP8 "
                    "--log_dir /work/build/logs/2025.07.22-17.34.36/B200-SXM-180GBx8_TRT/deepseek-r1/Offline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (e.g., DeepSeek_R1)")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario (e.g., Offline, Server)")
    parser.add_argument("--precision", type=str, required=True, help="Precision (e.g., FP8, FP16, FP32)")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to log directory")
    args = parser.parse_args()

    # Map string arguments to enums/constants
    try:
        benchmark = getattr(C.Benchmark, args.benchmark)
    except AttributeError:
        logging.error(f"Unknown benchmark: {args.benchmark}")
        sys.exit(1)
    try:
        scenario = getattr(C.Scenario, args.scenario)
    except AttributeError:
        logging.error(f"Unknown scenario: {args.scenario}")
        sys.exit(1)
    try:
        precision = getattr(C.Precision, args.precision)
    except AttributeError:
        logging.error(f"Unknown precision: {args.precision}")
        sys.exit(1)

    system = DETECTED_SYSTEM

    # Create a Workload instance
    wl = Workload(benchmark=benchmark,
                  scenario=scenario,
                  system=system)
    wl.log_dir = Path(args.log_dir)

    # Run accuracy check
    result = check_accuracy(wl)
    print("Accuracy check result:")
    print(result)
