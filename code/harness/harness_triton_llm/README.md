# MLPerf Triton LLM harness.

# Table of Contents
- [Glossary](#glossary)
- [Design](#design)
- [Components](#components)
- [Backend](#backend)
  - [Throughput limitations of gRPC](#throughput-limitations-of-grpc)
  - [The solution](#the-solution)
  - [Example](#example)
- [Frontend](#frontend)
- [TLDR, please just give me a config](#tldr-please-just-give-me-a-config)
- [Steps to run](#steps-to-run)
- [Multi-node usage](#multi-node-usage)
- [Important note on config override](#important-note-on-config-override)
- [Known issues](#known-issues)

## Glossary
- TP: Tensor Parallelism
- PP: Pipeline Parallelism
- DP: Data Parallelism
- gRPC channel: A single TCP connection, which corresponds to a single pair of client and server TCP ports

## Design
The MLPerf Inference Triton harness is designed for LLM workloads - Llama2-70B, Mixtral-8x7b and Llama3.1-405B.  
This doc will explain, in brief, the design of the Triton harness. 

## Components
The entire benchmarking setup has the following components:
1. Loadgen: This is the MLPerf Inference loadgen, that issues queries and accepts responses
2. `tritonclient` processes: This is a set of processes that initializes Triton's [gRPC client](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/_client.py). This is capable of sending LLM queries, and receiving streamed or non-streamed responses, from the `tritonserver`. We shall refer to the client processes as the _Frontends_
3. `tritonserver` processes: This is a set of processes that runs one or multiple [Triton server](https://github.com/triton-inference-server/server) processes, essentially spawning TRTLLM GptManager(s), that is, running one or more instances of an LLM engine and exposing a gRPC endpoint to send requests to and receive responses from those LLM engines. These processes are refered to as the _Backend_

## Backend
The backend may comprise one, or more, `tritonserver` processes. We use the [`launch_triton_server.py`](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/scripts/launch_triton_server.py) script from Triton's TensorRT-LLM backend to spawn the tritonserver process. A single `tritonserver` process exposes all the models it loads by a single gRPC server. If a single gRPC port is unable to sustain cumulative throughput, we divide models between `tritonserver` instances (i.e., between different gRPC ports).

### Throughput limitations of gRPC
To understand the throughput limitations of gRPC, refer to [the gRPC performance doc](https://grpc.io/docs/guides/performance/) (limit on concurrent streams), and the [open issue on github](https://github.com/grpc/grpc/issues/21386).  

TLDR: There is a limit on the number of active streaming RPCs (100 per channel). This becomes a problem in server scenario - every pending sample is an active stream of tokens on the gRPC channel, and if this reaches beyond 100 (usually the case in llama2-70B and mixtral-8x7b), it will bottleneck the responses.  

(gRPC channel: a pair of client-server TCP ports)

### The solution
As recommended in the gRPC performance document, we utilize multiple channels. This implies we possibly need multiple gRPC ports on the server side, hence, multiple `tritonserver` processes.

### Example
Taking Llama2-70B on H200x8 as an example, where we run 8 instances of llama2-70b (DP=8) with TP=1 and PP=1.

For Offline mode, a single `tritonserver` process is spawned which loads 8 instances (DP=8) of TP=1 PP=1 TRTLLM engines. A single `tritonserver` process means a single gRPC port is shared. This is enough, since the tokens are not streamed back, thus number of concurrent RPCs does not exceed the threshold.

However, for server mode, we launch 8 `tritonserver` processes, each loading a single TP=1, PP=1 instance of llama2-70b on a single GPU. We now ping 8 gRPC ports for request/response from the client side. Per-channel, the concurrent stream threshold is not crossed.

**The number of tritonserver instances is controlled via the `triton_num_servers` config field. For Offline scenario, we can keep this to 1. For server scenario, please keep this equal to the number of DP ranks spawned on that host**

For example, if you are running llama2-70b on H100x8 on DP=4, TP=1, PP=2, please specify `triton_num_servers = 4` in the Server configuration. 

For Offline, `triton_num_servers = 1` should be used in all cases

## Frontend
The frontend comprises of the processes owning the `tritonclient.grpc.client` instances. This process takes in samples from the loadgen, formats it as required by the triton gRPC client, and sends it for asynchronous inference to the server.

A single `tritonclient.grpc.client` can not sustain the throughput of the samples for obvious reasons. Hence, we need multiple clients. These clients can be split across threads in the same process space (using `threading.Thread`) or split across different prcess spaces (using `multiprocessing.Process`). The two configuration variables controlling this are:

- `triton_num_frontends_per_model`: Number of distinct `multiprocessing.Process` workers for each model instance. Call this a `frontend`
- `triton_num_clients_per_frontend`: Number of distinct `threading.Thread` workers in each `frontend` above. Each thread owns a single `tritonclient.grpc.client` object for asynchronous inference.

Clients belonging to the same `frontend` process share the gRPC channel (i.e., the ephemeral port on the client side), while clients belonging to different `frontend` processes will have separate gRPC channels, i.e., separate ephemeral client ports.

For Offline, it is usually enough to set both of these as 1. For server, careful tuning is required, but you are welcome to use the values in our configuration files. 

## TLDR, please just give me a config
If the details above were too verbose, please use the following skeleton config:

```
@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_Triton_PP2x1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.YourSystemName
    
    # Required for Triton harness
    use_triton = True
    
    # triton_num_clients_per_frontend: 1 for Offline, (2, 4 or 8) for Server
    triton_num_clients_per_frontend = 1

    # triton_num_frontends_per_model: 1 for Offline, (2, 4 or 8) for Server
    triton_num_frontends_per_model = 1

    # triton_num_servers: 1 for Offline, num_dp_ranks for Server
    triton_num_servers = 1

    # Rest of the config below
    gpu_batch_size = {'model-name': 2048}
    offline_expected_qps = 25
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
    }
    trtllm_runtime_flags = {'max_num_tokens': 1024}
```

## Steps to run
0. Enter the container and install TRTLLM. From `closed/NVIDIA`:
```
make prebuild
```
Then, from inside container:

```
make clone_trt_llm && make build_trt_llm
```


1. Run the following steps to install the required Triton software.

```
make clone_triton && make build_triton
```
- This should be required only once, as all compilations are binary files, not required if you re-enter the container

2. Generate the engines.

You may skip this part if you have an engine already built that you want to reuse - use the `--engine_dir` flag in `run_harness` step.
```
make generate_engines RUN_ARGS="--benchmarks=llama2 --scenarios=Offline[,Server] --harness_type=triton --accuracy_target=.999"
make generate_engines RUN_ARGS="--benchmarks=moe --scenarios=Offline[,Server] --harness_type=triton"
```
3. Generate Triton config files

To generate the triton model repo with config.pbtxt, use:
```
make generate_triton_config RUN_ARGS="--benchmarks=<model_name> --scenarios=<server/offline> --engine_dir=path/to/trtllm/engine --harness_type=triton"
```

This will use the runtime and build time parameters from your config (in configs/<model>/<scenario>) and set that in the config.pbtxt inside the model repo `/work/build/triton_model_repo_*` (one for each server dictated by `triton_num_servers`)

Please review the config.pbtxt to cross-check that it has the correct values.

4. Run the harness.

```
make run_harness RUN_ARGS="--benchmarks=llama2 --scenarios=Offline[,Server] --harness_type=triton --accuracy_target=.999"
make run_harness RUN_ARGS="--benchmarks=moe --scenarios=Offline[,Server] --harness_type=triton"
```

## Multi-node usage
Currently, multi-node support is only for DP ranks split across nodes. A single model may not run split between nodes. Multiple hosts will only multiply your DP ranks. Steps are as follows:
1. One node will be designated as the master node, while all other will be the slave nodes.
2. Get the hostname of the slave nodes via `hostname` linux command. 
3. Make sure the master node can reach each of the slave nodes over the network via `ping slave_hostname`.
4. Now, launch a tritonserver process on _every_ node, i.e., all the master nodes and the slave nodes. You may use the triton model repository created for you in the `make generate_triton_config` step.

A sample command using the launch_triton_server.py script will look like:
```
python3 /work/build/triton-inference-server/out/tensorrtllm/scripts/launch_triton_server.py \
  --tritonserver=/opt/tritonserver/bin/tritonserver \
  --model_repo=/work/build/triton_model_repo_0 \
  --tensorrt_llm_model_name=llama3_1-405b-offline-0 \
  --world_size=4 
```

5. Once the tritonserver processes are spawned on every node, launch the benchmark script as:
```
make run_harness RUN_ARGS="--models=<model-name> --scenarios=<scenario> --harness_type=triton --triton_skip_server_spawn --triton_grpc_ports=host_0:port_0,port_1|host_1:port_0,port_1"
```
The `triton_grpc_ports` value must have, for each host, comma separated port values (`hostname:port_0,port_1`) (multiple ports come from possible multiple servers). Then, combine this for each host delimited by the `|` character.

An example with 2 nodes named `host_foo` and `host_bar`, in Offline, where you may spawn a single tritonserver process on port `8001` and `8003` respectively, is `--triton_grpc_ports=host_foo:8001|host_bar:8003` 

## Important note on config override
The TRTLLM_RUNTIME_FLAGS **will not work** with the triton harness currently - please do perf tuning on the default python harness, and hard-code those parameters in the config files for the Triton configuration under the respective `configs/` folder.

TRTLLM_BUILD_FLAGS and TRTLLM_CHECKPOINT_FLAGS should work as usual with `make generate_engines`. 

For more info, check the code/harness/harness_llm_py/README.md

## Known issues
#### must specify `--harness_type=triton` in `make generate_triton_config`
In make generate_triton_config step, if you do not specify `--harness_type=triton` to the RUN_ARGS, it will fail.

