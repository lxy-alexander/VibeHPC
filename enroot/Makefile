ifndef MAKEFILE_ENROOT_INCLUDED
MAKEFILE_ENROOT_INCLUDED := 1

### Repo root directory
repo_root_path = $(shell git rev-parse --show-toplevel)


### Module paths
trtllm_module_path = $(repo_root_path)/closed/NVIDIA/3rdparty/trtllm
inference_module_path = $(repo_root_path)/closed/NVIDIA/3rdparty/mlc-inference
mitten_module_path = $(repo_root_path)/closed/NVIDIA/3rdparty/mitten


### Module compiling commands
TRTLLM_BUILD_ARCH ?= 100-real;103-real
TRTLLM_COMPILE_CMD ?= python3 scripts/build_wheel.py --use_ccache --benchmarks -a='$(TRTLLM_BUILD_ARCH)' --no-venv --clean

### Targets

## Submodule handling
.PHONY: update_submodules update_trtllm update_inference update_mitten

update_submodules: update_trtllm update_inference update_mitten

update_module:
	@if [ -z "$(module_path)" ]; then \
		echo "module_path must not be empty"; \
		exit 1; \
	fi
	git submodule update --init --recursive $(module_path)

update_trtllm:
	@$(MAKE) update_module module_path=$(trtllm_module_path)

update_inference:
	@$(MAKE) update_module module_path=$(inference_module_path)

update_mitten:
	@$(MAKE) update_module module_path=$(mitten_module_path)

## Submodule building
.PHONY: build_trtllm_devel compile_trtllm build_trtllm_rel build_mlperf_rel

build_trtllm_devel:
	@echo "\033[0;33mWarning: Make sure this command is run from within a slurm allocation\033[0m"
	@if [ -z "$(TRTLLM_DEVEL_SQSH)" ]; then \
		echo "\033[0;31mPlease specify a sqsh file via TRTLLM_DEVEL_SQSH=... to save trtllm_devel image to\033[0m"; \
		exit 1; \
	elif [ -f "$(TRTLLM_DEVEL_SQSH)" ]; then \
		echo "\033[0;33m$(TRTLLM_DEVEL_SQSH) exists\033[0m - assuming this is fully built sqsh file and skipping sqsh build"; \
		echo " >>> If you wish to force rebuild the trtllm devel image, pls rm $(TRTLLM_DEVEL_SQSH) or provide a new sqsh path"; \
		echo "\n"; \
	else \
		cd $(trtllm_module_path) && make -C enroot build_sqsh SQSH_PATH=$(TRTLLM_DEVEL_SQSH); \
	fi;

compile_trtllm:
	@if ls $(trtllm_module_path)/build/tensorrt_llm*.whl 1>/dev/null 2>&1; then \
		echo "\033[0;33mwhl exists for TRTLLM\033[0m - skipping compile_trtllm"; \
		echo "\n"; \
	else \
		$(MAKE) build_trtllm_devel; \
		$(MAKE) run_srun_step \
			CONTAINER_IMAGE=$(TRTLLM_DEVEL_SQSH) \
			CONTAINER_WORKDIR=$(trtllm_module_path) \
			CONTAINER_MOUNTS="$(repo_root_path):$(repo_root_path)" \
			RUN_CMD="$(TRTLLM_COMPILE_CMD)"; \
	fi;

build_trtllm_rel:
	@if [ -z "$(TRTLLM_REL_SQSH)" ]; then \
		echo "\033[0;31mPlease specify a rel sqsh file via TRTLLM_REL_SQSH=... to save trtllm_rel image to\033[0m"; \
		exit 1; \
	elif [ -f "$(TRTLLM_REL_SQSH)" ]; then \
		echo "\033[0;33m$(TRTLLM_REL_SQSH) exists\033[0m - assuming this is fully built sqsh file and skipping sqsh build"; \
		echo " >>> If you wish to force rebuild the trtllm rel image, pls rm $(TRTLLM_REL_SQSH) or provide a new sqsh path"; \
		echo "\n"; \
	else \
		$(MAKE) compile_trtllm; \
		$(MAKE) build_sqsh \
			CONTAINER_IMAGE=$(TRTLLM_DEVEL_SQSH) \
			CONTAINER_SAVE=$(TRTLLM_REL_SQSH) \
			CONTAINER_MOUNTS="$(repo_root_path):$(repo_root_path)" \
			CONTAINER_WORKDIR=$(trtllm_module_path) \
			RUN_CMD="pip install $(trtllm_module_path)/build/tensorrt_llm*.whl"; \
	fi;

build_mlperf_rel:
	@if [ -z "$(MLPERF_REL_SQSH)" ]; then \
		echo "\033[0;31mPlease specify a rel sqsh file via MLPERF_REL_SQSH=... to save trtllm_rel image to\033[0m"; \
		exit 1; \
	elif [ -f "$(MLPERF_REL_SQSH)" ]; then \
		echo "\033[0;33m$(MLPERF_REL_SQSH) exists\033[0m - assuming this is fully built sqsh file and skipping sqsh build"; \
		echo " >>> If you wish to force rebuild the trtllm rel image, pls rm $(MLPERF_REL_SQSH) or provide a new sqsh path"; \
		echo "\n"; \
	else \
		$(MAKE) build_trtllm_rel; \
		$(MAKE) build_sqsh \
			CONTAINER_IMAGE=$(TRTLLM_REL_SQSH) \
			CONTAINER_SAVE=$(MLPERF_REL_SQSH) \
			CONTAINER_MOUNTS="$(repo_root_path):$(repo_root_path)" \
			CONTAINER_WORKDIR=$(repo_root_path)/closed/NVIDIA \
			RUN_CMD="pip install --user $(inference_module_path)/loadgen $(mitten_module_path) -r docker/common/requirements/requirements.llm.txt"; \
	fi;


build_sqsh:
	@if [ -z "$(CONTAINER_SAVE)" ]; then \
		echo "\033[0;31mPlease specify a output sqsh file via CONTAINER_SAVE=... to build sqsh\033[0m"; \
		exit 1; \
	fi
	@$(MAKE) run_srun_step EXTRA_SRUN_FLAGS="--container-save $(CONTAINER_SAVE) $(EXTRA_SRUN_FLAGS)"

run_srun_step:
	@if [ -z "$(CONTAINER_IMAGE)" ]; then \
		echo "\033[0;31mPlease specify a sqsh file via CONTAINER_IMAGE=... to run srun step\033[0m"; \
		exit 1; \
	fi
	@if [ -z "$(RUN_CMD)" ]; then \
		echo "\033[0;31mPlease specify a run command via RUN_CMD=... to run srun step\033[0m"; \
		exit 1; \
	fi
	@if [ -z "$(CONTAINER_WORKDIR)" ]; then \
		echo "\033[0;31mPlease specify a code directory via CONTAINER_WORKDIR=... to run srun step\033[0m"; \
		exit 1; \
	fi
	@if [ -z "$(CONTAINER_MOUNTS)" ]; then \
		echo "\033[0;31mPlease specify a container mounts via CONTAINER_MOUNTS=... to run srun step\033[0m"; \
		exit 1; \
	fi

	@echo "Running srun pyxis step."
	@echo "    input sqsh image: \033[0;31m$(CONTAINER_IMAGE)\033[0m"
	@echo "    mounts: \033[0;31m$(CONTAINER_MOUNTS)\033[0m"
	@echo "    home directory: \033[0;31m$(CONTAINER_WORKDIR)\033[0m"
	@echo "    run command: \033[0;31m$(RUN_CMD)\033[0m"

	srun --container-remap-root \
		--container-image $(CONTAINER_IMAGE) \
		--container-workdir $(CONTAINER_WORKDIR) \
		--container-mounts $(CONTAINER_MOUNTS) \
		$(EXTRA_SRUN_FLAGS) \
		$(RUN_CMD)

endif
