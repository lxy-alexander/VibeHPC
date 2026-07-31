#!/usr/bin/env python3
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


__doc__ = """NVIDIA's MLPerf Inference Benchmark submission code. NVIDIA's implementation runs in 2 phases.

The first phase is 'engine generation', which builds a TensorRT Engine using TensorRT, a Deep Learning Inference
performance optimization SDK by NVIDIA. This only applies to NVIDIA accelerator-based workloads.

The second phase is a 'harness run', which launches the generated TensorRT engine in a server-like harness that
accepts input from LoadGen (MLPerf Inference's official Load Generator), runs the inference with the engine, and reports
the output back to LoadGen.

More about the MLPerf Inference Benchmark and NVIDIA's submission implementation can be found in the README.md for this
project.
"""
from code import G_BENCHMARK_MODULES
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import atexit
import sys
from typing import List, Optional, Tuple

from nvmitten.configurator import (
    Configuration,
    ConfigurationIndex,
    Field,
    HelpInfo,
    autoconfigure,
    bind,
)
from nvmitten.importer import ScopedImporter
from nvmitten.pipeline import Pipeline, ScratchSpace
from nvmitten.system.system import System

import code.common.constants as C
import code.common.paths as paths
import code.fields.gen_engines as builder_fields
import code.fields.harness as harness_fields
from code.fields.harness import MPIMode
import code.fields.meta as metafields
import code.fields.general as general_fields
import code.fields.loadgen as lg_fields
import code.ops as Ops

from code.common import logging
from code.common.power_limit import get_power_context
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.workload import Workload
from code.common.mlcommons.compliance import get_audit_verifier, set_audit_conf
from code.llmlib.config import HarnessConfig, TrtllmEndpointConfig, TrtllmHlApiConfig
import dataclasses


@autoconfigure
@bind(metafields.action)
@bind(metafields.benchmarks)
@bind(metafields.scenarios)
@bind(metafields.harness_type)
@bind(metafields.accuracy_target)
@bind(metafields.power_setting)
@bind(general_fields.config_dir)
@bind(harness_fields.audit_test)
@bind(harness_fields.mpi_mode)
@bind(harness_fields.config_id)
@bind(general_fields.show_help, "show_help")
@bind(general_fields.verbose)
@bind(general_fields.verbose_nvsmi)
@bind(general_fields.log_dir)
@bind(lg_fields.test_mode)
class MainRunner:
    def __init__(self,
                 system: System,
                 action: C.Action = None,
                 benchmarks: List[C.Benchmark] = None,
                 scenarios: List[C.Scenario] = None,
                 harness_type: Optional[C.HarnessType] = None,
                 accuracy_target: C.AccuracyTarget = C.AccuracyTarget(0.99),
                 power_setting: C.PowerSetting = C.PowerSetting.MaxP,
                 show_help: bool = False,
                 config_dir: os.PathLike = paths.WORKING_DIR / "configs",
                 audit_test: Optional[C.AuditTest] = None,
                 verbose: bool = False,
                 verbose_nvsmi: bool = False,
                 log_dir: os.PathLike = paths.BUILD_DIR / "logs" / "default",
                 mpi_mode: MPIMode = MPIMode.LEGACY,
                 config_id: str = 'default',
                 test_mode: str = 'PerformanceOnly'):
        assert action is not None, "No action specified"
        assert benchmarks is not None, "No benchmarks specified"
        assert scenarios is not None, "No scenarios specified"

        self.system = system
        self.system_id = system.extras["id"]
        self.action = action
        self.benchmarks = benchmarks
        self.scenarios = scenarios

        self.harness_type = harness_type
        self.accuracy_target = accuracy_target
        self.power_setting = power_setting

        self.config_dir = config_dir
        self.audit_test = audit_test
        self.mpi_mode = mpi_mode
        self.config_id = config_id
        self.test_mode = test_mode

        self.show_help = show_help
        self.verbose = verbose
        self.verbose_nvsmi = verbose_nvsmi
        self.log_dir = log_dir
        self.nvidia_smi_process = None
        self.nvidia_smi_csv_file = None

        self.config_index = ConfigurationIndex()
        for benchmark in self.benchmarks:
            for scenario in self.scenarios:
                p = self.conf_path(benchmark, scenario)
                if not p.exists():
                    logging.info(f"Config file {p} not found. Loading minimal configs as default.")
                    _subdir = "minimal"
                else:
                    logging.info(f"Loading configs from {p}")
                    _subdir = self.system_id

                with ScopedImporter([str(self.config_dir / _subdir)] + sys.path):
                    imp_path = f"{scenario.valstr}.{benchmark.valstr}"
                    self.config_index.load_module(imp_path, prefix=[_subdir, benchmark, scenario])

    def _start_nvidia_smi_monitoring(self):
        """Start nvidia-smi monitoring if verbose_nvsmi is enabled."""
        if not self.verbose_nvsmi:
            return

        # Get polling interval from environment variable (default: 200ms)
        polling_interval_ms = int(os.environ.get('NVSMI_REFRESH_RATE', 200))

        # Import nvidia_smi_csv_keys from the GPU fields script
        scripts_path = paths.WORKING_DIR / "scripts" / "perf_monitor"
        sys.path.insert(0, str(scripts_path))
        try:
            from nvsmi_gpu_fields import nvidia_smi_csv_keys
            all_fields = list(nvidia_smi_csv_keys.keys())
        except ImportError:
            logging.warning("Could not import nvidia-smi GPU fields script, using default fields")
            all_fields = ["timestamp", "pci.bus_id", "power.draw", "utilization.gpu", "temperature.gpu"]
        finally:
            if str(scripts_path) in sys.path:
                sys.path.remove(str(scripts_path))

        # Create log directory if it doesn't exist
        log_dir_path = Path(self.log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Prepare nvidia-smi command
        csv_file = log_dir_path / "nvidia_smi_monitor.csv"
        self.nvidia_smi_csv_file = csv_file  # Store for later plotting
        cmd = [
            "nvidia-smi",
            "--format=csv",
            f"--loop-ms={polling_interval_ms}",
            f"--filename={csv_file}",
            f"--query-gpu={','.join(all_fields)}"
        ]

        try:
            logging.info(f"Starting nvidia-smi monitoring with {polling_interval_ms}ms polling interval, output: {csv_file}")
            self.nvidia_smi_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            atexit.register(self._stop_nvidia_smi_monitoring)
        except Exception as e:
            logging.warning(f"Failed to start nvidia-smi monitoring: {e}")
            self.nvidia_smi_process = None

    def _stop_nvidia_smi_monitoring(self):
        """Stop nvidia-smi monitoring process and generate plots."""
        if self.nvidia_smi_process is not None:
            try:
                logging.info("Stopping nvidia-smi monitoring...")
                self.nvidia_smi_process.terminate()
                # Wait a bit for graceful termination
                try:
                    timeout = int(os.getenv('NVSMI_KILL_TIMEOUT', 5))
                    self.nvidia_smi_process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logging.warning("nvidia-smi process did not terminate gracefully, killing it")
                    self.nvidia_smi_process.kill()
                self.nvidia_smi_process = None
            except Exception as e:
                logging.warning(f"Error stopping nvidia-smi monitoring: {e}")

        # Generate plots if CSV file was created
        if self.verbose_nvsmi and self.nvidia_smi_csv_file is not None:
            self._generate_nvsmi_plots()

    def _generate_nvsmi_plots(self):
        """Generate plots from nvidia-smi CSV data."""
        if not self.nvidia_smi_csv_file.exists():
            logging.warning(f"nvidia-smi CSV file not found: {self.nvidia_smi_csv_file}")
            return

        # Give nvidia-smi time to flush data
        import time
        time.sleep(2)

        # Check if CSV file has content
        try:
            file_size = self.nvidia_smi_csv_file.stat().st_size
            if file_size == 0:
                logging.warning(f"nvidia-smi CSV file is empty: {self.nvidia_smi_csv_file}")
                return

            # Validate CSV has data rows (not just headers)
            with open(self.nvidia_smi_csv_file, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:
                logging.warning(f"nvidia-smi CSV file has insufficient data (only {len(lines)} lines): {self.nvidia_smi_csv_file}")
                return

            # Check for GPU identifier columns
            header_line = lines[0].strip().lower()
            has_gpu_identifier = any(col in header_line for col in ['uuid', 'index'])
            if not has_gpu_identifier:
                logging.warning(f"nvidia-smi CSV file missing GPU identifier columns. Header: {lines[0][:200]}")
                return

            logging.info(f"nvidia-smi CSV file has {len(lines)} lines, proceeding with plot generation")

        except Exception as e:
            logging.warning(f"Error validating nvidia-smi CSV file: {e}")
            return

        try:
            # Install plotting requirements first
            logging.info("Installing plotting requirements...")
            plot_requirements = paths.PROJECT_BASE_DIR / "scripts" / "plot" / "requirements.txt"
            install_cmd = f"python3 -m pip install -q -r {plot_requirements}"
            subprocess.run(install_cmd, shell=True, check=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Prepare output directory
            log_dir_path = Path(self.log_dir)
            output_dir = log_dir_path / "nvsmi_plots"

            # Run plotting script
            logging.info(f"Generating nvidia-smi plots from {self.nvidia_smi_csv_file}...")
            plot_script = paths.PROJECT_BASE_DIR / "scripts" / "plot" / "nvsmi_csv.py"
            plot_cmd = [
                "python3",
                str(plot_script),
                str(self.nvidia_smi_csv_file),
                "-o", str(output_dir),
            ]

            result = subprocess.run(plot_cmd, check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)

            logging.info(f"nvidia-smi plots generated successfully in {output_dir}")
            if result.stdout:
                logging.debug(f"Plot generation output:\n{result.stdout}")

        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to generate nvidia-smi plots: {e}")
            if e.stderr:
                logging.warning(f"Error output: {e.stderr}")
        except Exception as e:
            logging.warning(f"Error generating nvidia-smi plots: {e}")

    def conf_path(self, benchmark: C.Benchmark, scenario: C.Scenario) -> Path:
        """Get the path to the configuration file for a given benchmark and scenario.

        Args:
            benchmark (C.Benchmark): The benchmark to get the config path for
            scenario (C.Scenario): The scenario to get the config path for

        Returns:
            Path: The path to the configuration file
        """
        return self.config_dir / self.system_id / scenario.valstr / f"{benchmark.valstr}.py"

    def _get_atomic_config(self, benchmark: C.Benchmark, scenario: C.Scenario,
                           workload_setting: C.WorkloadSetting) -> dict:
        """Load atomic config for the given benchmark/scenario/workload_setting and config_id.

        Args:
            benchmark (C.Benchmark): The benchmark to load config for
            scenario (C.Scenario): The scenario to load config for
            workload_setting (C.WorkloadSetting): The workload setting to load config for

        Returns:
            dict: The selected atomic config

        Raises:
            RuntimeError: If atomic config cannot be loaded or found
        """
        # Determine which subdir to load from (system_id or minimal)
        p = self.conf_path(benchmark, scenario)
        _subdir = self.system_id if p.exists() else "minimal"

        # Import the module to access ATOMIC_EXPORTS
        with ScopedImporter([str(self.config_dir / _subdir)] + sys.path):
            imp_path = f"{scenario.valstr}.{benchmark.valstr}"
            module = __import__(imp_path, fromlist=['ATOMIC_EXPORTS'])

            # Access ATOMIC_EXPORTS - will raise AttributeError if missing
            try:
                atomic_exports = module.ATOMIC_EXPORTS
            except AttributeError:
                raise RuntimeError(f"No ATOMIC_EXPORTS found in {imp_path}")

            # Access workload setting - will raise KeyError if missing
            try:
                atomic_configs = atomic_exports[workload_setting]
            except KeyError:
                available_settings = list(atomic_exports.keys())
                raise RuntimeError(
                    f"WorkloadSetting '{workload_setting.short}' not found in ATOMIC_EXPORTS. "
                    f"Available settings: {available_settings}"
                )

            # Validate structure
            if not isinstance(atomic_configs, dict) or not all(isinstance(k, str) for k in atomic_configs.keys()):
                raise RuntimeError(f"ATOMIC_EXPORTS[{workload_setting}] must be a dict with string keys (config_ids)")

            # Access config_id - will raise KeyError if missing
            try:
                config = atomic_configs[self.config_id]
            except KeyError:
                available_ids = list(atomic_configs.keys())
                raise RuntimeError(
                    f"config_id '{self.config_id}' not found in ATOMIC_EXPORTS. "
                    f"Available config_ids: {available_ids}"
                )

            logging.info(f"Using atomic config with config_id='{self.config_id}'")
            return config

    @staticmethod
    def _deep_merge(base: dict, overrides: dict) -> dict:
        """Deep merge overrides into base dict.

        For nested dicts, recursively merge. For other types, override replaces base.
        """
        result = base.copy()
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = MainRunner._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _load_accuracy_overrides(self, benchmark: C.Benchmark, scenario: C.Scenario,
                                  workload_setting: C.WorkloadSetting) -> Optional[dict]:
        """Load ACCURACY_OVERRIDES from config module if available.

        Args:
            benchmark: The benchmark to load overrides for
            scenario: The scenario to load overrides for
            workload_setting: The workload setting to load overrides for

        Returns:
            dict of overrides if found, None otherwise
        """
        p = self.conf_path(benchmark, scenario)
        _subdir = self.system_id if p.exists() else "minimal"

        with ScopedImporter([str(self.config_dir / _subdir)] + sys.path):
            imp_path = f"{scenario.valstr}.{benchmark.valstr}"
            try:
                module = __import__(imp_path, fromlist=['ACCURACY_OVERRIDES'])
                accuracy_overrides = getattr(module, 'ACCURACY_OVERRIDES', None)
                if accuracy_overrides is None:
                    return None
                overrides = accuracy_overrides.get(workload_setting)
                if overrides:
                    logging.info(f"Loaded ACCURACY_OVERRIDES for {benchmark.valstr}/{scenario.valstr}")
                return overrides
            except (ImportError, AttributeError):
                return None

    def _load_compliance_overrides(self, benchmark: C.Benchmark, scenario: C.Scenario,
                                   workload_setting: C.WorkloadSetting,
                                   audit_test: C.AuditTest) -> Optional[dict]:
        """Load COMPLIANCE_OVERRIDES from config module if available.

        Args:
            benchmark: The benchmark to load overrides for
            scenario: The scenario to load overrides for
            workload_setting: The workload setting to load overrides for
            audit_test: The audit test to load overrides for

        Returns:
            dict of overrides if found, None otherwise
        """
        p = self.conf_path(benchmark, scenario)
        _subdir = self.system_id if p.exists() else "minimal"

        with ScopedImporter([str(self.config_dir / _subdir)] + sys.path):
            imp_path = f"{scenario.valstr}.{benchmark.valstr}"
            try:
                module = __import__(imp_path, fromlist=['COMPLIANCE_OVERRIDES'])
                compliance_overrides = getattr(module, 'COMPLIANCE_OVERRIDES', None)
                if compliance_overrides is None:
                    return None
                # Look up by audit_test first, then by workload_setting
                if audit_test not in compliance_overrides:
                    return None
                test_overrides = compliance_overrides[audit_test]
                if workload_setting not in test_overrides:
                    return None
                logging.info(f"Found COMPLIANCE_OVERRIDES for {audit_test.valstr}")
                return test_overrides[workload_setting]
            except (ImportError, AttributeError):
                return None

    def _run_workload(self, benchmark: C.Benchmark, scenario: C.Scenario):
        """Run a specific workload for a given benchmark and scenario.

        This method sets up the workload configuration, creates a pipeline, and executes it
        with the appropriate power context.

        Args:
            benchmark (C.Benchmark): The benchmark to run
            scenario (C.Scenario): The scenario to run the benchmark under
        """
        # Override log directory for audit tests
        if self.audit_test is not None:
            # Check if we even need to run the audit test in the first place
            verifier = get_audit_verifier(self.audit_test)
            if benchmark in verifier.exclude_list:
                logging.info(f"Skipping audit test {self.audit_test.valstr} for {benchmark.valstr} {scenario.valstr} as it is not needed for submission.")
                return

            _audit_log_dir = paths.BUILD_DIR / "compliance_logs" / self.audit_test.valstr
            _audit_log_dir.mkdir(parents=True, exist_ok=True)
            os.environ["LOG_DIR"] = str(_audit_log_dir)

        # self.audit_test == None will cleanup any audit configs before harness starts
        set_audit_conf(self.audit_test, benchmark)

        ht = benchmark.default_harness_type if self.harness_type is None else self.harness_type
        workload_setting = C.WorkloadSetting(harness_type=ht,
                                             accuracy_target=self.accuracy_target,
                                             power_setting=self.power_setting)

        # Get config based on mode and action
        # ATOMIC_EXPORTS: Used for run_llm_server in leader mode (atomic system configs)
        #                 Also used for run_harness/run_audit_harness with dynamo_cluster config
        # EXPORTS: Used for run_harness with default config (harness system configs, may aggregate multiple atomic systems)

        # NOTE - Enabling ATOMIC_EXPORT usage for run_harness only for dynamo runs for now,
        # to allow dynamo have it's own harness fields (target_qps, etc) and not break non-dynamo workloads
        use_atomic_config = (
            (self.mpi_mode == MPIMode.LEADER and self.action == C.Action.RunLLMServer) or
            (self.config_id == 'dynamo_cluster' and self.action in (C.Action.RunHarness, C.Action.RunAuditHarness))
        )
        if use_atomic_config:
            # Leader mode server requires atomic configs - will raise error if not found
            atomic_config = self._get_atomic_config(benchmark, scenario, workload_setting)
            # Apply ACCURACY_OVERRIDES for atomic configs if test_mode is AccuracyOnly
            if self.test_mode == 'AccuracyOnly':
                overrides = self._load_accuracy_overrides(benchmark, scenario, workload_setting)
                if overrides:
                    for field, value in overrides.items():
                        if field in atomic_config and isinstance(atomic_config[field], dict) and isinstance(value, dict):
                            atomic_config[field] = self._deep_merge(atomic_config[field], value)
                            logging.info(f"Applying accuracy override (deep merge): {field.name}")
                        else:
                            atomic_config[field] = value
                            logging.info(f"Applying accuracy override: {field.name} = {value}")
            config = Configuration(atomic_config)
        else:
            # Non-leader mode uses standard EXPORTS
            keyspace = [self.system_id, benchmark, scenario, workload_setting]
            config = self.config_index.get(keyspace)
            if config is None:
                logging.warning(f"Config not found for current system. Attempting to load minimal config for {benchmark.valstr} {scenario.valstr} ({workload_setting.short})")
                config = self.config_index.get(["minimal", benchmark, scenario, workload_setting])
                if config is None:
                    logging.error("No minimal config found. Using empty config")
                    config = Configuration()

        # Apply overrides based on mode:
        # 1. COMPLIANCE_OVERRIDES - for specific audit tests that need custom datasets
        # 2. ACCURACY_OVERRIDES - for AccuracyOnly mode
        if not use_atomic_config:
            overrides = None

            # First, try compliance overrides for audit tests
            if self.audit_test is not None:
                overrides = self._load_compliance_overrides(benchmark, scenario, workload_setting, self.audit_test)

            # Fall back to accuracy overrides for AccuracyOnly mode (if no compliance overrides found)
            if overrides is None and self.test_mode == 'AccuracyOnly':
                overrides = self._load_accuracy_overrides(benchmark, scenario, workload_setting)

            if overrides:
                for field, value in overrides.items():
                    if field in config and isinstance(config[field], dict) and isinstance(value, dict):
                        config[field] = self._deep_merge(config[field], value)
                        logging.info(f"Applying override (deep merge): {field.name}")
                    else:
                        config[field] = value
                        logging.info(f"Applying override: {field.name} = {value}")

        # Sanitize configuration for types. Maybe this should be provided in Mitten?
        for k, v in config.items():
            assert isinstance(k, Field), f"Invalid Configuration key {k} is not a Mitten Field object"
            if isinstance(v, str) and (k.from_string and k.from_string is not str):
                logging.debug(f"Configuration - Parsing string for field {k.name}")
                config[k] = k.from_string(v)

        # Use .from_fields since there is no auto-applied config yet.
        wl = Workload.from_fields(benchmark,
                                  scenario,
                                  system=self.system,
                                  setting=workload_setting)
        config[Workload.FIELD] = wl

        if self.action == C.Action.GenerateEngines:
            config[builder_fields.force_build_engines] = True

        power_context = get_power_context()

        with config.autoapply():
            if benchmark.is_llm and benchmark is not C.Benchmark.WHISPER:  # Whisper is using PyHarnessOp in run harness
                core_type = HarnessConfig().core_type

                if self.action in (C.Action.GenerateTritonConfig,):
                    ops = self.get_triton_generate_config_op(benchmark)

                if self.action in (C.Action.GenerateDisaggConfig,):
                    ops = self.get_trtllm_disagg_generate_config_op(benchmark)

                if self.action in (C.Action.GenerateEngines,):
                    ops = self.get_llm_generate_engine_ops(benchmark, core_type)

                if self.action in (C.Action.RunLLMServer,):
                    ops = self.get_llm_launch_server_ops(benchmark, core_type)

                if self.action in (C.Action.RunHarness,):
                    ops = self.get_llm_harness_run_ops(benchmark, core_type)

            else:
                ops = self.get_harness_run_ops(benchmark)

            if self.show_help:
                print(HelpInfo.build_help_string(ops))
                sys.exit(0)

            scratch_space = ScratchSpace(paths.BUILD_DIR)
            pipeline = Pipeline(scratch_space, ops, dict())

            # Start nvidia-smi monitoring if verbose_nvsmi is enabled
            self._start_nvidia_smi_monitoring()

            with power_context:
                pipeline.run()

    def get_llm_generate_engine_ops(self, benchmark: C.Benchmark, core_type: Optional[harness_fields.CoreType] = None):
        """
        Get list of operations to generate LLM engines for given --core_type
        This will build one or more engines as needed by the workload.
        """

        def get_build_ops(pipeline: Tuple[str] = ()):
            m = G_BENCHMARK_MODULES[benchmark]
            m.load(pipeline)

            impls = m.custom_op_impls
            ops = []

            for k in list(pipeline):
                if impls[k] is not None:
                    ops.append(impls[k])

            if self.show_help and impls["EngineBuilderOp"]:
                for builder in m.component_map.values():
                    if builder:
                        for c in builder.mro():
                            HelpInfo.add_configurator_dependency(impls["EngineBuilderOp"], c)

            return ops

        match core_type:
            case harness_fields.CoreType.TRITON_GRPC: ops = get_build_ops(("CalibrateEngineOp", "EngineBuilderOp", "GenerateTritonConfigOp",))
            case harness_fields.CoreType.TRTLLM_EXECUTOR: ops = get_build_ops(("CalibrateEngineOp", "EngineBuilderOp", ))
            case harness_fields.CoreType.TRTLLM_DISAGG: ops = get_build_ops(("HFQuantizerOp", ))
            case harness_fields.CoreType.TRTLLM_ENDPOINT:
                requires_engine = TrtllmEndpointConfig().runtime_flags['trtllm_backend'] == 'cpp'
                ops = get_build_ops(("HFQuantizerOp",) if not requires_engine else ("CalibrateEngineOp", "EngineBuilderOp", ))
            case harness_fields.CoreType.TRTLLM_HLAPI:
                requires_engine = TrtllmHlApiConfig().runtime_flags['trtllm_backend'] == 'cpp'
                ops = get_build_ops(("HFQuantizerOp",) if not requires_engine else ("CalibrateEngineOp", "EngineBuilderOp", ))
            case harness_fields.CoreType.DISAGG_FRONTEND:
                # Frontend doesn't need any engine/model
                ops = []
            case harness_fields.CoreType.DISAGG_PREFILL | harness_fields.CoreType.DISAGG_DECODE:
                # Prefill and decode workers need quantized checkpoint
                ops = get_build_ops(("HFQuantizerOp",))
            case harness_fields.CoreType.DYNAMO_ENDPOINT:
                # Dynamo endpoint is harness-only mode - no build needed
                ops = []
            case _: raise NotImplementedError(f"Unsupported core type: {core_type}")
        return ops

    def get_llm_launch_server_ops(self, benchmark: C.Benchmark, core_type: Optional[harness_fields.CoreType] = None):
        """
        Get list of operations to launch LLM Servers for given core_type
        This will launch one or multiple LLM Servers to expose benchmark-able endpoints.
        This will also run calibration and generate engines if needed.
        """

        def get_launch_ops(additional_ops: Tuple[str] = ()):
            ops = self.get_llm_generate_engine_ops(benchmark, core_type)

            m = G_BENCHMARK_MODULES[benchmark]
            m.load(additional_ops)
            for k in additional_ops:
                if m.custom_op_impls[k] is not None:
                    ops.append(m.custom_op_impls[k])

            return ops

        match core_type:
            case harness_fields.CoreType.TRITON_GRPC: ops = get_launch_ops(("GenerateTritonConfigOp", "RunTritonServerOp"))
            case harness_fields.CoreType.TRTLLM_ENDPOINT: ops = self.get_llm_generate_engine_ops(benchmark, core_type) + get_launch_ops(("RunTrtllmServeOp",))
            case harness_fields.CoreType.TRTLLM_DISAGG: ops = get_launch_ops(("RunTrtllmServeDisaggOp",))
            case harness_fields.CoreType.TRTLLM_EXECUTOR: ops = []  # no server
            case harness_fields.CoreType.TRTLLM_HLAPI: ops = []  # no server
            case harness_fields.CoreType.DISAGG_FRONTEND:
                from code.llmlib.launch_server import RunDisaggFrontendOp
                ops = [RunDisaggFrontendOp]
            case harness_fields.CoreType.DISAGG_PREFILL:
                from code.llmlib.launch_server import RunDisaggPrefillOp
                # Prefill workers need HFQuantizerOp for model
                ops = self.get_llm_generate_engine_ops(benchmark, core_type) + [RunDisaggPrefillOp]
            case harness_fields.CoreType.DISAGG_DECODE:
                from code.llmlib.launch_server import RunDisaggDecodeOp
                # Decode workers need HFQuantizerOp for model
                ops = self.get_llm_generate_engine_ops(benchmark, core_type) + [RunDisaggDecodeOp]
            case harness_fields.CoreType.DYNAMO_ENDPOINT:
                # Dynamo endpoint is harness-only mode - server already running
                ops = []
            case _: raise NotImplementedError(f"Unsupported core type: {core_type}")
        return ops

    def get_llm_harness_run_ops(self, benchmark: C.Benchmark, core_type: Optional[harness_fields.CoreType] = None):
        """Get the list of operations to be performed for a given llm benchmark.

        The operations list varies depending on the core_type being used.

        Args:
            benchmark (C.Benchmark): The benchmark to get operations for
        """
        def get_run_ops(backend_modules: Tuple[str] = ()):
            m = G_BENCHMARK_MODULES[benchmark]
            m.load(backend_modules)
            impls = {
                "LoadgenConfFilesOp": Ops.LoadgenConfFilesOp,
                "ResultSummaryOp": Ops.ResultSummaryOp,
            }
            impls |= m.custom_op_impls

            ops = []
            pipeline_steps = ["LoadgenConfFilesOp"] + list(backend_modules) + ["ResultSummaryOp"]
            for k in pipeline_steps:
                if impls[k] is not None:
                    ops.append(impls[k])

            return ops

        match core_type:
            case harness_fields.CoreType.TRTLLM_EXECUTOR: ops = get_run_ops(("TrtllmExecutorBenchmarkHarnessOp",))
            case harness_fields.CoreType.TRTLLM_ENDPOINT: ops = get_run_ops(("TrtllmServeBenchmarkHarnessOp",))
            case harness_fields.CoreType.TRTLLM_DISAGG: ops = get_run_ops(("TrtllmDisaggServeBenchmarkHarnessOp",))
            case harness_fields.CoreType.TRITON_GRPC: ops = get_run_ops(("TritonBenchmarkHarnessOp",))
            case harness_fields.CoreType.TRTLLM_HLAPI: ops = get_run_ops(("TrtllmHLApiBenchmarkHarnessOp",))
            case harness_fields.CoreType.DISAGG_FRONTEND | harness_fields.CoreType.DISAGG_PREFILL | harness_fields.CoreType.DISAGG_DECODE:
                # Disaggregated serving exposes OpenAI-compatible HTTP API via frontend
                ops = get_run_ops(("TrtllmServeBenchmarkHarnessOp",))
            case harness_fields.CoreType.DYNAMO_ENDPOINT:
                # Dynamo endpoint uses minimal harness with no engine dependencies
                ops = get_run_ops(("DynamoEndpointBenchmarkHarnessOp",))
            case _: raise NotImplementedError(f"Unsupported core type: {core_type}")

        # NOTE(vir):
        # add engine generation steps in pipeline to satisfy dependency outputs
        # engine build ops do not overwrite existing engines by default
        ops = self.get_llm_generate_engine_ops(benchmark, core_type) + ops
        return ops

    def get_harness_run_ops(self, benchmark: C.Benchmark):
        """Get the list of operations to be performed for a given benchmark.

        The operations list varies depending on the action type (GenerateEngines or RunHarness)
        and includes operations like calibration, engine building, and result summarization.

        Args:
            benchmark (C.Benchmark): The benchmark to get operations for

        Returns:
            list: List of operations to be performed for the benchmark
        """
        m = G_BENCHMARK_MODULES[benchmark]
        m.load()
        impls = {"CalibrateEngineOp": Ops.CalibrateEngineOp,
                 "EngineBuilderOp": Ops.EngineBuilderOp,
                 "BenchmarkHarnessOp": Ops.BenchmarkHarnessOp,
                 "ResultSummaryOp": Ops.ResultSummaryOp}
        impls |= m.custom_op_impls

        ops = []
        for k in ("CalibrateEngineOp", "EngineBuilderOp"):
            if impls[k] is not None:
                ops.append(impls[k])

        if self.show_help and impls["EngineBuilderOp"]:
            for builder in m.component_map.values():
                if builder:
                    for c in builder.mro():
                        HelpInfo.add_configurator_dependency(impls["EngineBuilderOp"], c)

        if self.action == C.Action.GenerateEngines:
            return ops

        ops.append(Ops.LoadgenConfFilesOp)
        for k in ("BenchmarkHarnessOp", "ResultSummaryOp"):
            if impls[k] is not None:
                ops.append(impls[k])

        return ops

    def get_triton_generate_config_op(self, benchmark: C.Benchmark):
        """Get the list of operations to generate Triton Server config files.
        """
        m = G_BENCHMARK_MODULES[benchmark]
        m.load(("GenerateTritonConfigOp", ))
        impls = {"CalibrateEngineOp": Ops.CalibrateEngineOp,
                 "EngineBuilderOp": Ops.EngineBuilderOp}
        impls |= m.custom_op_impls

        ops = []

        for k in ("CalibrateEngineOp", "EngineBuilderOp", "GenerateTritonConfigOp"):
            if impls[k] is not None:
                ops.append(impls[k])

        return ops

    def get_trtllm_disagg_generate_config_op(self, benchmark: C.Benchmark):
        """Get the list of operations to generate Trtllm Disagg Server config files.
        """
        m = G_BENCHMARK_MODULES[benchmark]
        m.load(("HFQuantizerOp", "GenerateTrtllmDisaggConfigOp",))
        impls = m.custom_op_impls
        ops = []

        for k in ("HFQuantizerOp", "GenerateTrtllmDisaggConfigOp"):
            if impls[k] is not None:
                ops.append(impls[k])

        return ops

    def run_all(self):
        """Run all configured workloads.

        This method iterates through all configured benchmarks and scenarios,
        running each workload in sequence.
        """
        for benchmark in self.benchmarks:
            for scenario in self.scenarios:
                self._run_workload(benchmark, scenario)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    Ops.MPS().disable()
    if "id" not in DETECTED_SYSTEM.extras:
        logging.info(f"Detected system did not match any known systems. Exiting. {DETECTED_SYSTEM}")
    else:
        logging.info(f"Detected system ID: {DETECTED_SYSTEM.extras['id']}")
        with Configuration().autoapply():  # Create empty Configuration to invoke autoconfigure
            runner = MainRunner(DETECTED_SYSTEM)
        runner.run_all()
