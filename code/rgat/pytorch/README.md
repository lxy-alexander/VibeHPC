# RGAT (GNN) Benchmark

This benchmark performs node classifications on the IGB-Heterogeneous (Full) dataset.


## Model

The model is quite tiny, and is a 3-layer RGAT classification network.

### Downloading

To download the model, run `./download_model.sh`


## Dataset

:warning: This dataset is extremely large and requires multiple preprocessing steps with large intermediate files. You
will need to obtain a separate scratch space with ~10 TB to have enough storage to also contain intermediate files.

Mount this scratch space in the docker container using `DOCKER_ARGS="-v path/to/scratch:/home/mlperf_inf_rgat"`


### Downloading

To download the dataset, run `./download_igbh_full.sh`


### Preprocessing

To preprocess the dataset, run `./preprocess_data.sh`

### Running the model

There are special dependencies required for RGAT, and you need to launch the docker using the following command

```
make prebuild INSTALL_RGAT_DEPS=1
make run_harness RUN_ARGS="--benchmarks=rgat --scenarios=Offline --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=rgat --scenarios=Offline --test_mode=PerformanceOnly"
```

There is no audit tests for RGAT
