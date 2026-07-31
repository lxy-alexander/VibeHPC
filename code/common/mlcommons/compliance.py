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
from typing import Optional
import math
import os
import shutil
import json
import sys

from nvmitten.configurator import bind, autoconfigure

from code.common import logging, run_command
from code.common import paths
from code.common.workload import Workload
from code.common.constants import AuditTest, Benchmark

from .accuracy_checker import check_accuracy
from .venv_utils import ensure_venv_ready

from pathlib import Path


def _move_file(src, dst):
    logging.info(f"=> Compliance harness: Moving file: {src} --> {dst}")
    shutil.move(src, dst)


def _copy_file(src, dst):
    logging.info(f"=> Compliance harness: Copying file: {src} --> {dst}")
    shutil.copy(src, dst)


def set_audit_conf(audit_test: Optional[AuditTest] = None,
                   benchmark: Optional[Benchmark] = None):
    """Set the audit configuration for the given audit test."""
    audit_files = ["audit.config",
                   "verify_accuracy.txt",
                   "verify_performance.txt",
                   "mlperf_log_accuracy_baseline.json",
                   "accuracy.txt",
                   "predictions.json"]

    for fname in audit_files:
        if os.path.exists(fname):
            logging.info(f"Cleaning up leftover audit test file - {fname}")
            os.remove(fname)

    if audit_test is None:
        return
    assert benchmark is not None, "Must specify a benchmark if audit_test is specified"

    benchmark_name = benchmark.valstr
    if benchmark_name == "gptj":
        benchmark_name = "gpt-j"
    elif benchmark_name == "wan22-a14b":
        benchmark_name = "wan-2.2-t2v-a14b"

    src_config = paths.MLCOMMONS_INF_REPO / 'compliance' / audit_test.valstr / benchmark_name / 'audit.config'
    logging.info(f'AUDIT HARNESS: Looking for audit.config in {src_config}...')
    if not src_config.is_file():
        # For tests that have one central audit.config instead of per-benchmark
        src_config = paths.MLCOMMONS_INF_REPO / 'compliance' / audit_test.valstr / 'audit.config'
        logging.info(f'AUDIT HARNESS: Search failed. Looking for audit.config in {src_config}...')
        assert src_config.is_file(), f"Audit config file {src_config} not found"

    # Destination is audit.config
    dest_config = 'audit.config'
    # Copy the file
    shutil.copyfile(str(src_config), str(dest_config))


@autoconfigure
@bind(Workload.FIELD)
class AuditVerifier:
    """Base class for audit verifiers."""

    exclude_list = []

    def __init__(self,
                 test_id: AuditTest,
                 script_path: os.PathLike,
                 workload: Optional[Workload] = None):
        self.test_id = test_id
        self.script_path = script_path
        self.submitter = os.environ.get("SUBMITTER", "NVIDIA")

        assert workload is not None, "Must specify a workload. Is autoconfigure working?"
        self.workload = workload
        self.raw_log_dir = workload.log_dir

        self.results_path = paths.RESULTS_STAGING_DIR / \
            "closed" / \
            self.submitter / \
            "results" / \
            self.workload.submission_system / \
            self.workload.submission_benchmark / \
            self.workload.scenario.valstr

        if not self.results_path.exists():
            raise FileNotFoundError(f"Results path {self.results_path} does not exist. Have you run `make stage_results`?")

        self.flags = []

    def run(self) -> str:
        """Runs the audit verification script and returns the command output if a 0 exit code is returned."""
        if self.workload.benchmark in self.__class__.exclude_list:
            return f"{self.test_id.valstr}_PASS_EXEMPTED"

        flag_str = " ".join(self.flags)

        cmd = f"{sys.executable} {self.script_path} {flag_str}"
        return run_command(cmd, get_output=True)


class AuditTest01Verifier(AuditVerifier):
    """Verifier for TEST01."""

    exclude_list = [
        Benchmark.GPTJ,
        Benchmark.LLAMA2,
        Benchmark.LLAMA3_1_8B,
        Benchmark.LLAMA3_1_405B,
        Benchmark.Mixtral8x7B,
        Benchmark.DeepSeek_R1,
        Benchmark.GPT_OSS_120B,
        Benchmark.WAN22_A14B,
    ]

    def __init__(self):
        super().__init__(
            AuditTest.TEST01,
            paths.MLCOMMONS_INF_REPO / "compliance" / "TEST01" / "run_verification.py")

        self.mlperf_accuracy_log = self.results_path / "accuracy" / "mlperf_log_accuracy.json"
        if not self.mlperf_accuracy_log.exists():
            raise FileNotFoundError(f"MLPerf accuracy log {self.mlperf_accuracy_log} does not exist. Did you run --test_mode=AccuracyOnly before `make stage_results`?")

        with self.mlperf_accuracy_log.open() as f:
            try:
                json.load(f)  # Checks for JSONDecodeError
            except json.JSONDecodeError as e:
                raise ValueError(f"MLPerf accuracy log {self.mlperf_accuracy_log} is malformed. Did you already truncate?\n{e}")

        self.flags = [f"--results={self.results_path}",
                      f"--compliance={self.raw_log_dir}",
                      f"--output_dir={self.raw_log_dir}"]

    def run(self) -> str:
        try:
            result = super().run()
            if "Accuracy check pass: True" in result and "Performance check pass: True" in result:
                logging.info(f"=> Compliance test TEST01 passed without fallback")
                return "TEST01_PASS_NORMAL"
            else:
                # Raise exception since fallback is handled in `except` block
                raise ValueError("Accuracy checker script failed. Proceeding to fallback approach.")
        except:
            logging.info("=> Running fallback for TEST01")

            # 1. Generate baseline_accuracy file
            baseline_script_path = paths.MLCOMMONS_INF_REPO / "compliance" / "TEST01" / "create_accuracy_baseline.sh"
            fallback_command = f"bash {baseline_script_path} {self.mlperf_accuracy_log} {self.raw_log_dir / 'mlperf_log_accuracy.json'}"
            run_command(fallback_command)
            baseline_acc_log = Path("mlperf_log_accuracy_baseline.json")
            assert baseline_acc_log.exists(), f"Baseline accuracy log {baseline_acc_log} does not exist. Baseline script failed."

            # 2. Create accuracy and performance directories
            accuracy_dir = self.raw_log_dir / "TEST01" / "accuracy"
            accuracy_dir.mkdir(parents=True, exist_ok=True)

            performance_dir = self.raw_log_dir / "TEST01" / "performance" / "run_1"
            performance_dir.mkdir(parents=True, exist_ok=True)

            # 3. Calculate the accuracy of baseline, using the benchmark's accuracy script
            self.workload.audit_test01_fallback_mode = True
            fallback_result_baseline_list = check_accuracy(self.workload)
            assert Path("accuracy.txt").exists(), "Accuracy checker script failed - Did not generate accuracy.txt"
            _move_file('accuracy.txt', str(accuracy_dir / 'baseline_accuracy.txt'))

            # 4. Calculate the accuracy of the compliance run, using the benchmark's accuracy script
            self.workload.audit_test01_fallback_mode = False
            fallback_result_list = check_accuracy(self.workload)

            # 5. Copy all the compliance files into log directory
            # Move it to the submission dir - check_accuracy stores accuracy.txt in the directory
            # name provided in its first argument. So this file will already be located inside get_full_log_dir()
            _move_file(str(self.raw_log_dir / 'accuracy.txt'),
                       str(accuracy_dir / 'compliance_accuracy.txt'))
            # Required for run_verification.py
            # Move the required logs to their correct locations since run_verification.py has failed.
            _move_file('verify_accuracy.txt', str(self.raw_log_dir / 'TEST01' / 'verify_accuracy.txt'))
            _copy_file(str(self.raw_log_dir / 'mlperf_log_accuracy.json'), str(accuracy_dir / 'mlperf_log_accuracy.json'))
            _copy_file(str(self.raw_log_dir / 'mlperf_log_detail.txt'), str(performance_dir / 'mlperf_log_detail.txt'))
            _copy_file(str(self.raw_log_dir / 'mlperf_log_summary.txt'), str(performance_dir / 'mlperf_log_summary.txt'))

            # 6. Check whether or not the accuracies are within a defined tolerance
            verify_performance_script = paths.MLCOMMONS_INF_REPO / "compliance" / "TEST01" / "verify_performance.py"
            verify_performance_args = f"-r {self.results_path / 'performance' / 'run_1' / 'mlperf_log_detail.txt'} -t {performance_dir / 'mlperf_log_detail.txt'}"
            verify_performance_command = f"{self.workload.benchmark.python_path} {verify_performance_script} {verify_performance_args} | tee {self.raw_log_dir / 'TEST01' / 'verify_performance.txt'}"
            logging.info(f"Running verify command: {verify_performance_command}")
            run_command(verify_performance_command)

            acc_target = self.workload.setting.accuracy_target.value
            rel_tol = 1.0 - acc_target
            logging.info(f"=> Compliance harness: Detected accuracy target of {acc_target * 100:.1f}%")
            logging.info(f"=> Compliance harness: Tolerance set to {rel_tol * 100:.1f}%")

            for i, fallback_result_baseline in enumerate(fallback_result_baseline_list):
                acc_metric = fallback_result_baseline["name"]
                if acc_metric == "GEN_LEN":
                    # Special case for GPT-J GEN_LEN, which needs to be within 10%
                    rel_tol = 0.1

                baseline_acc_value = fallback_result_baseline["value"]
                compliance_acc_value = fallback_result_list[i]["value"]
                if not (_fb_metric := fallback_result_list[i]["name"]) == acc_metric:
                    logging.error(f"=> Compliance harness: Metric mismatch: BASELINE METRIC: {acc_metric} COMPLIANCE METRIC: {_fb_metric}")
                    return "TEST01_FAIL_METRIC_MISMATCH"

                if not math.isclose(baseline_acc_value, compliance_acc_value, rel_tol=rel_tol):
                    logging.error(f'=> Compliance harness: RelTol failure in {acc_metric}: BASELINE ACCURACY: {baseline_acc_value}, COMPLIANCE_ACCURACY: {compliance_acc_value}')
                    return "TEST01_FAIL_FALLBACK_RELTOL"

            logging.info('AUDIT HARNESS: Success: TEST01 failure redeemed via fallback approach.')
            return "TEST01_PASS_FALLBACK"


class AuditTest04Verifier(AuditVerifier):
    """Verifier for TEST04."""

    exclude_list = [
        Benchmark.BERT,
        Benchmark.DLRMv2,
        Benchmark.Retinanet,
        Benchmark.GPTJ,
        Benchmark.LLAMA2,
        Benchmark.LLAMA3_1_8B,
        Benchmark.LLAMA3_1_405B,
        Benchmark.Mixtral8x7B,
        Benchmark.RGAT,
        Benchmark.WHISPER,
        Benchmark.DeepSeek_R1,
        Benchmark.GPT_OSS_120B,
    ]

    def __init__(self):
        super().__init__(
            AuditTest.TEST04,
            paths.MLCOMMONS_INF_REPO / "compliance" / "TEST04" / "run_verification.py")

        self.flags = [f"--results_dir={self.results_path}",
                      f"--compliance_dir={self.raw_log_dir}",
                      f"--output_dir={self.raw_log_dir}"]

    def run(self) -> str:
        result = super().run()
        if "Performance check pass: True" in result:
            return "TEST04_PASS"
        else:
            return "TEST04_FAIL"


class AuditTest06Verifier(AuditVerifier):
    """Verifier for TEST06."""

    # TEST06 is for a subset of LLM benchmarks, so invert the list
    exclude_list = [
        b
        for b in Benchmark
        if b not in {
            Benchmark.LLAMA2,
            Benchmark.LLAMA3_1_8B,
            Benchmark.LLAMA3_1_405B,
            Benchmark.Mixtral8x7B,
            Benchmark.DeepSeek_R1,
        }]

    def __init__(self):
        super().__init__(
            AuditTest.TEST06,
            paths.MLCOMMONS_INF_REPO / "compliance" / "TEST06" / "run_verification.py")

        self.flags = [f"--compliance_dir={self.raw_log_dir}",
                      f"--output_dir={self.raw_log_dir}",
                      f"--scenario={self.workload.scenario.valstr}",
                      f"--dtype=int32"]

    def run(self) -> str:
        result = super().run()
        first_token_pass = "First token check pass: True" in result or "First token check pass: Skipped" in result
        eos_pass = "EOS check pass: True" in result
        length_check_pass = "Sample length check pass: True" in result

        if not first_token_pass:
            return "TEST06_FAIL_FIRST_TOKEN"
        elif not eos_pass:
            return "TEST06_FAIL_EOS"
        elif not length_check_pass:
            return "TEST06_FAIL_LENGTH"
        else:
            return "TEST06_PASS"


class AuditTest07Verifier(AuditVerifier):
    """Verifier for TEST07."""

    # Only gpt-oss-120b has TEST07 config
    exclude_list = [b for b in Benchmark if b != Benchmark.GPT_OSS_120B]

    def __init__(self):
        super().__init__(
            AuditTest.TEST07,
            paths.MLCOMMONS_INF_REPO / "compliance" / "TEST07" / "run_verification.py")

        # Paths for accuracy evaluation
        accuracy_script_path = paths.MLCOMMONS_INF_REPO / "language" / "gpt-oss-120b" / "eval_mlperf_accuracy.py"
        ref_data_path = paths.BUILD_DIR / "data" / "gpt-oss" / "v4" / "acc" / "acc_eval_compliance_gpqa.parquet"
        audit_config_path = paths.MLCOMMONS_INF_REPO / "compliance" / "TEST07" / "gpt-oss-120b" / "audit.config"

        # Build accuracy script command with {accuracy_log} placeholder
        # Use the gpt-oss accuracy venv (same as GptOss120bAccuracyChecker)
        self.venv_path = Path("/work/.gptoss-acc-venv")
        requirements_file = paths.CODE_DIR / "gpt-oss-120b" / "tensorrt" / "requirements.accuracy.txt"
        self.venv_path = ensure_venv_ready(self.venv_path, requirements_file)

        venv_python = self.venv_path / "bin" / "python3"
        accuracy_script_cmd = (
            f"{venv_python} {accuracy_script_path} "
            f"--mlperf-log {{accuracy_log}} "
            f"--reference-data {ref_data_path}"
        )

        self.flags = [
            f"--compliance_dir={self.raw_log_dir}",
            f"--output_dir={self.raw_log_dir}",
            f"--accuracy-script \"{accuracy_script_cmd}\"",
            f"--audit-config={audit_config_path}",
        ]

    def run(self) -> str:
        result = super().run()
        # Convert list to string if needed (run_command may return a list of lines)
        if isinstance(result, list):
            result_str = '\n'.join(result)
        else:
            result_str = result
        # TEST07/run_verification.py prints "Accuracy check pass: True/False" to stdout
        # but writes "TEST PASS"/"TEST FAIL" only to verify_accuracy.txt (not stdout).
        # run_command captures stdout, so check for the stdout string.
        return "TEST07_PASS" if "Accuracy check pass: True" in result_str else "TEST07_FAIL"


class AuditTest09Verifier(AuditVerifier):
    """Verifier for TEST09."""

    # Only gpt-oss-120b has TEST09 config
    exclude_list = [b for b in Benchmark if b != Benchmark.GPT_OSS_120B]

    def __init__(self):
        super().__init__(
            AuditTest.TEST09,
            paths.MLCOMMONS_INF_REPO / "compliance" / "TEST09" / "run_verification.py")

        audit_config_path = paths.MLCOMMONS_INF_REPO / "compliance" / "TEST09" / "gpt-oss-120b" / "audit.config"

        self.flags = [
            f"--compliance_dir={self.raw_log_dir}",
            f"--output_dir={self.raw_log_dir}",
            f"--audit-config={audit_config_path}",
        ]

    def run(self) -> str:
        result = super().run()
        # Convert list to string if needed (run_command may return a list of lines)
        if isinstance(result, list):
            result_str = '\n'.join(result)
        else:
            result_str = result
        return "TEST09_PASS" if "TEST PASS" in result_str else "TEST09_FAIL"


def get_audit_verifier(audit_test: AuditTest):
    """Get the audit verifier for the given audit test."""
    return {
        AuditTest.TEST01: AuditTest01Verifier,
        AuditTest.TEST04: AuditTest04Verifier,
        AuditTest.TEST06: AuditTest06Verifier,
        AuditTest.TEST07: AuditTest07Verifier,
        AuditTest.TEST09: AuditTest09Verifier,
    }[audit_test]
