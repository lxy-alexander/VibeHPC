# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is the main entrypoint of NVIDIA's MLPerf Inference codebase, and includes all of the other Makefiles used
# in the project.
# This file contains the targets used to run the actual MLPerf Inference workloads, as well as some basic utility
# commands.

MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))  # Must be declared before includes
include $(CURDIR)/Makefile.const
include $(CURDIR)/Makefile.docker
include $(CURDIR)/Makefile.build
include $(CURDIR)/Makefile.data
include $(CURDIR)/Makefile.tests
include $(CURDIR)/Makefile.submission
include $(CURDIR)/Makefile.pyxis

# Generate TensorRT engines (plan files) and run the harness.
.PHONY: run
run:
	@$(MAKE) -f $(MAKEFILE_NAME) generate_engines
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness

# Generate TensorRT engines (plan files).
.PHONY: generate_engines
generate_engines: link_dirs
	@$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="generate_engines"

.PHONY: generate_conf_files
generate_conf_files:
	@$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="generate_conf_files"

.PHONY: generate_triton_config
generate_triton_config:
	@$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="generate_triton_config"

.PHONY: generate_disagg_config
generate_disagg_config:
	@$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="generate_disagg_config"

.PHONY: run_llm_server
run_llm_server:
	@$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="run_llm_server"

# Run the harness and check accuracy if in AccuracyOnly mode.
# Add "set -o pipefail" so that "<command> | tee output.txt" will fail when "<command>" fails because otherwise tee
# would return status 0 and clear the failure status.
NO_DISPLAY_RESULTS ?= 0
IGNORE_ACCURACY_FAILURE ?= 0
MPI_CMD ?= mpirun
MPI_EXPORTS ?= -x LD_LIBRARY_PATH
MPI_ARGS ?= --allow-run-as-root
.PHONY: run_harness
run_harness: link_dirs
	@benchmark=$$(echo "$(RUN_ARGS)" | tr '[:upper:]' '[:lower:]' | sed -n 's/.*--benchmark[ =]\([^ ]*\).*/\1/p'); \
	benchmarks=$$(echo "$(RUN_ARGS)" | tr '[:upper:]' '[:lower:]' | sed -n 's/.*--benchmarks[ =]\([^ ]*\).*/\1/p'); \
	if echo "$$benchmark $$benchmarks" | grep -qE "q3vl|qwen3-vl-235b-a22b"; then \
		clean_args=$$(echo "$(RUN_ARGS)" | sed -E 's/--benchmark[ =][^ ]*//g' | sed -E 's/--benchmarks[ =][^ ]*//g'); \
		$(MAKE) -f $(MAKEFILE_NAME) run_harness_q3vl RUN_ARGS="$$clean_args --benchmarks=qwen3-vl-235b-a22b"; \
	else \
		mkdir -p $(LOG_DIR); \
		set -o pipefail && LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="run_harness" 2>&1 | tee $(LOG_DIR)/stdout.txt; \
	fi
ifeq ($(NO_DISPLAY_RESULTS), 0)
	@$(MAKE) -f $(MAKEFILE_NAME) display_results
endif

.PHONY: run_harness_q3vl
run_harness_q3vl: link_dirs
	@mkdir -p $(LOG_DIR)
	@bash -c 'set -o pipefail; \
		LD_LIBRARY_PATH="$(HARNESS_LD_LIBRARY_PATH)"; \
		if [ -n "$${SLURM_JOB_ID:-}" ] || [ -n "$${SLURM_STEP_ID:-}" ]; then \
			echo "Detected Slurm env: SLURM_JOB_ID=$${SLURM_JOB_ID:-} SLURM_STEP_ID=$${SLURM_STEP_ID:-}"; \
			echo "Q3VL harness command: LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action=\"run_harness\""; \
			$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="run_harness"; \
		else \
			DEFAULT_MPI_NP=$$(nvidia-smi -L 2>/dev/null | wc -l | tr -d " "); \
			MPI_NP=$${SLURM_NTASKS:-$${MPI_NP:-$${DEFAULT_MPI_NP:-1}}}; \
			echo "Q3VL harness command: LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(MPI_CMD) $(MPI_ARGS) $(MPI_EXPORTS) -np $$MPI_NP $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action=\"run_harness\""; \
			$(MPI_CMD) $(MPI_ARGS) $(MPI_EXPORTS) -np $$MPI_NP \
				$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="run_harness"; \
		fi' \
		2>&1 | tee $(LOG_DIR)/stdout.txt
ifeq ($(NO_DISPLAY_RESULTS), 0)
	@if [ -z "$$OMPI_COMM_WORLD_RANK" ] || [ "$$OMPI_COMM_WORLD_RANK" = "0" ]; then \
		$(MAKE) -f $(MAKEFILE_NAME) display_results; \
	fi
endif

.PHONY: display_results
display_results:
	@set -o pipefail && $(PYTHON3_CMD) -m scripts.print_harness_result --log_dir=$(LOG_DIR) --ignore-accuracy-failure=$(IGNORE_ACCURACY_FAILURE) 2>&1 | tee -a $(LOG_DIR)/stdout.txt

# Run the harness and measure power consumption using official MLPerf-Inference power workflow.
# Use environment var USE_WIN_PTD=1 to control using windows power server or not
.PHONY: run_harness_power
run_harness_power: link_dirs
ifeq ($(IS_SOC), 0)
	@echo "WARNING: If you have not already, before you run MaxQ mode, `make power_set_maxq_state` must be run *OUTSIDE* the docker container."
endif
	@mkdir -p $(POWER_LOGS_DIR)
	@$(MAKE) -f Makefile.power power_prologue
	set -o pipefail && $(PYTHON3_CMD) $(POWER_CLIENT_SCRIPT) -p $(POWER_SERVER_PORT) -a $(POWER_SERVER_IP) -n "$(POWER_NTP_URL)" -S \
		-L $(POWER_LOGS_TEMP_DIR) -o $(POWER_LOGS_DIR) -f \
		-w 'LOG_DIR=$(POWER_LOGS_TEMP_DIR) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="run_harness" \
			2>&1 | tee -a $(POWER_LOGS_DIR)/stdout.txt \
			&& if [ ! -d $(POWER_LOGS_DIR)/ranging_tmp ]; \
				then mkdir $(POWER_LOGS_DIR)/ranging_tmp \
					&& mv $(POWER_LOGS_TEMP_DIR)/* $(POWER_LOGS_DIR)/ranging_tmp/ \
					&& cp -v $(POWER_LOGS_DIR)/ranging_tmp/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
					&& cp -v $(POWER_LOGS_DIR)/ranging_tmp/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/; \
				else mkdir $(POWER_LOGS_DIR)/testing_tmp \
					&& mv $(POWER_LOGS_TEMP_DIR)/* $(POWER_LOGS_DIR)/testing_tmp/ \
					&& cp -v $(POWER_LOGS_DIR)/testing_tmp/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
					&& cp -v $(POWER_LOGS_DIR)/testing_tmp/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/; fi' \
	|| $(MAKE) -f Makefile.power power_kill_server
	@$(MAKE) -f Makefile.power power_epilogue

.PHONY: run_audit_harness
run_audit_harness: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test01
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test04
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test06
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test07
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test09

.PHONY: run_audit_test01_once
run_audit_test01_once: link_dirs
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --audit_test=TEST01 --server_target_qps_adj_factor=0.92 --action="run_harness"

# TEST01 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test01
run_audit_test01:
	@for i in 1 2; do echo "TEST01 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test01_once && break; done

.PHONY: run_audit_test04_once
run_audit_test04_once: link_dirs
	@echo "Sleep to reset thermal state before TEST04..." && sleep 20
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --audit_test=TEST04 --server_target_qps_adj_factor=0.92 --action="run_harness"

# TEST04 is so short that it is sometimes unstable. Try up to three times before failing.
.PHONY: run_audit_test04
run_audit_test04:
	@for i in 1 2 3; do echo "TEST04 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test04_once && break; done

.PHONY: run_audit_test06
run_audit_test06: link_dirs
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --audit_test=TEST06 --server_target_qps_adj_factor=0.92 --action="run_harness"

.PHONY: run_audit_test07
run_audit_test07: link_dirs
	@echo "Running TEST07..."
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --audit_test=TEST07 --server_target_qps_adj_factor=0.92 --action="run_harness"

.PHONY: run_audit_test09
run_audit_test09: link_dirs
	@echo "Running TEST09..."
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --audit_test=TEST09 --server_target_qps_adj_factor=0.92 --action="run_harness"

.PHONY: run_harness_correlation
run_harness_correlation: link_dirs
	@$(MAKE) -f Makefile.power lock_clocks
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness || ($(MAKE) -f Makefile.power free_clocks ; exit 1)
	@$(MAKE) -f Makefile.power free_clocks

# Re-generate TensorRT calibration cache.
.PHONY: calibrate
calibrate: link_dirs
	@$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="calibrate"


############################## UTILITY ##############################

# When running DLRM benchmark on multi-GPU systems, sometimes it is required to set this to reduce the likelihood of
# out-of-memory issues.
.PHONY: set_virtual_memory_overcommit
set_virtual_memory_overcommit:
	@sudo sysctl -w vm.max_map_count=16777216
	@sudo sysctl -w vm.overcommit_memory=2
	@sudo sysctl -w vm.overcommit_ratio=70


.PHONY: copy_profiles
copy_profiles:
	@echo "Copying yml and sqlite files"
	@cp *.yml /home/scratch.dlsim/data/mlperf-inference/
	@cp *.sqlite /home/scratch.dlsim/data/mlperf-inference/


.PHONY: autotune
autotune:
	@$(PYTHON3_CMD) -m scripts.internal.autotune.grid $(benchmark) $(scenario) $(config) $(RUN_ARGS)


# Remove build directory.
.PHONY: clean
clean: clean_shallow
	rm -rf build


# Remove only the files necessary for a clean build.
.PHONY: clean_shallow
clean_shallow:
	rm -rf build/bin
	rm -rf build/harness
	rm -rf build/plugins
	rm -rf $(TRTLLM_DIR)
	rm -rf $(TRITON_OUT_DIR)
	rm -rf $(LOADGEN_LIB_DIR)
	rm -rf $(TRITON_PREBUILT_LIBS_DIR) # Triton CPU libraries
	rm -rf openvino*


.PHONY: clean_triton
clean_triton:
	rm -rf $(TRITON_DIR)
	rm -rf $(TRITON_OUT_DIR)
	rm -rf $(TRITON_PREBUILT_LIBS_DIR) # Triton CPU libraries


# Print out useful information.
.PHONY: info
info:
	@echo "==== MLPerf:"
	@echo "VERSION=$(VERSION)"
	@echo "PARTNER_DROP=$(PARTNER_DROP)"
	@echo "PARTNER_RELEASE=$(PARTNER_RELEASE)"
	@echo "==== System:"
	@echo "RUN_ID=$(RUN_ID)"
	@echo "SYSTEM_NAME=$(shell $(PYTHON3_CMD) -m scripts.get_system_id 2> /dev/null)"
	@echo "Architecture=$(ARCH)"
	@echo "IS_SOC=$(IS_SOC)"
ifeq ($(IS_SOC), 1)
	@echo "SOC_SM=$(SOC_SM)"
else
	@echo "SM=$(SM)"
endif
	@echo "MIG_CONF=$(MIG_CONF)"
	@echo "==== Docker: "
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "HOSTNAME=$(HOSTNAME)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "UBUNTU_VERSION=$(UBUNTU_VERSION)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
	@echo "Docker file name: $(DOCKER_FILENAME)"
	@echo "Docker base image used: $(BASE_IMAGE_NAME)"
	@echo "Docker container registry used: $(CONTAINER_REGISTRY)"
	@echo "==== Env Vars:"
	@echo "PYTHON3_CMD=$(PYTHON3_CMD)"
	@echo "GIT_BIN"=$(GIT_BIN)
	@echo "PATH=$(PATH)"
	@echo "CPATH=$(CPATH)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "LIBRARY_PATH=$(LIBRARY_PATH)"
	@echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"
	@echo "HARNESS_LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH)"
	@echo "EXTERNAL_USER=$(EXTERNAL_USER)"
	@echo "==== SW:"
	@echo "NVCC_CUDA_VER=$(CUDA_VER), (Detected: $(shell nvcc --version | grep "release" | awk '{print $$6}' | cut -d V -f 2))"
	@echo "DRIVER_VER=$(DRIVER_VER)"
	@echo "DRIVER_VER_MAJOR=$(DRIVER_VER_MAJOR)"
	@echo "TRT_VER=$(TRT_VER), (Detected: $(shell python -c "import tensorrt; print(tensorrt.__version__)"))"
	@echo "CUDNN_VER=$(CUDNN_VER), (Detected: $(shell dpkg -l | grep libcudnn8-dev | awk '{print $$3}' | cut -d'-' -f 1))"
	@echo "TORCH_VER=(Detected: $(shell python -c "import torch; print(torch.__version__)"))"
	@echo "==== Build flags:"
	@echo "HARNESS_BUILD_FLAGS=$(HARNESS_BUILD_FLAGS)"
	@echo "TRITON_BUILD_FLAGS=$(TRITON_BUILD_FLAGS)"
	@echo "BUILD_TRITON=$(BUILD_TRITON)"
	@echo "USE_NIGHTLY=$(USE_NIGHTLY)"
	@echo "BUILD_TRTLLM=$(BUILD_TRTLLM)"



# The shell target will start a shell that inherits all the environment
# variables set by this Makefile for convenience.
.PHONY: shell
shell:
	@$(SHELL)


.PHONY: perf_profile
perf_profile:
	@if [ -z $(CSV_FILE) ]; then \
		echo "Error: You must specify CSV_FILE"; \
		exit 1; \
	fi
	@if [ -z $(IMG_FILE) ]; then \
		echo "Error: You must specify IMG_FILE"; \
		exit 1; \
	fi
	@if [ -z "$(COMMAND)" ]; then \
		if [ -z "$(ACTION)" ] || [ -z "$(RUN_ARGS)" ]; then \
			echo "Error: You must specify either COMMAND or both ACTION and RUN_ARGS"; \
			exit 1; \
		fi; \
		echo "Running with ACTION=$(ACTION) and RUN_ARGS=\"$(RUN_ARGS)\"" && \
		$(PYTHON3_CMD) scripts/perf_monitor/run_nvidia_smi_mon.py -o $(CSV_FILE) -c "make $(ACTION) RUN_ARGS=\"$(RUN_ARGS)\""; \
	else \
		echo "Running with COMMAND=$(COMMAND)"; \
		$(PYTHON3_CMD) scripts/perf_monitor/run_nvidia_smi_mon.py -o $(CSV_FILE) -c "$(COMMAND)"; \
	fi
	$(PYTHON3_CMD) scripts/perf_monitor/plot_nvidia_smi_mon.py -i $(CSV_FILE) -o $(IMG_FILE); \
