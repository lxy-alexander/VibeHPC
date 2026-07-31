# Bandwidth specs of datacenter systems for MLPerf Inference 6.0

## Introduction
As per the rules of MLPerf Inference, submissions are required to prove that the systems used provide a certain level of ingress (network to loadgen trace) and egress (loadgen trace to network) bandwidth.
Put simply, the throughput at which the accelerator accepts queries and generates responses should not exceed the maximum data bandwidth the system is capable of supporting.

Say the throughput of a benchmark run is X samples/second. It is necessary to document that, the system is capable of supporting at least X inputs/outputs from/to the network, per second.

In v6.0, NVIDIA submitted using the following datacenter systems:
- B200-SXM-180GBx8
- B300-SXM-288GBx8
- GB200-NVL72_GB200-186GB_aarch64x72
- GB300-NVL72_GB300-288GB_aarch64x72

## Calculating the maximum permissible QPS
For each workload, the offline scenario uses the most bandwidth. We list the bandwidth used per model `used_bw` (in byte/sec) for a run in Offline scenario.  
The `used_bw` is a function of throughput of input/output samples `tput` (sample/second) and size of each input/output sample `bytes/sample`.

### Ingress bandwidth requirements

| Benchmark                | Formula                                                                                                               | Bandwidth used   (bytes)        | Values                                                                                                               |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Llama3.1-8B              | ```used_bw = tput x max_input_len x dtype_size```                                                                     | ```used_bw = tput x 10240```    | ```max_input_len = 2560; dtype_size = 4B```                                                                          |
| Llama2-70B               | ```used_bw = tput x max_input_len x dtype_size```                                                                     | ```used_bw = tput x 4096```     | ```max_input_len = 1024; dtype_size = 4B```                                                                          |
| Llama3.1-405B            | ```used_bw = tput x max_input_len x dtype_size```                                                                     | ```used_bw = tput x 80000```    | ```max_input_len = 20000; dtype_size = 4B```                                                                         |
| DeepSeek-R1              | ```used_bw = tput x max_input_len x dtype_size```                                                                     | ```used_bw = tput x 12544```    | ```max_input_len = 3136; dtype_size = 4B```                                                                          |
| GPT-OSS-120B             | ```used_bw = tput x max_input_len x dtype_size```                                                                     | ```used_bw = tput x 61320```    | ```max_input_len = 15330; dtype_size = 4B```                                                                         |
| WAN-2.2-T2V-A14B         | ```used_bw = tput x max_prompt_len x dtype_size```                                                                    | ```used_bw = tput x 308```      | ```max_prompt_len = 77; dtype_size = 4B```                                                                           |
| Qwen3-VL-235B-A22B       | ```used_bw = tput x max_input_tokens x dtype_size```                                                                  | ```used_bw = tput x 108204```   | ```max_input_tokens = 27051; dtype_size = 4B```                                                                      |
| Whisper                  | ```used_bw = tput x audio_length x sample_rate x dtype_size```                                                        | ```used_bw = tput x 960000```   | ```max_audio_length = 30sec; sample_rate = 16000Hz; dtype_size = 2B```                                               |
| R-GAT                    | negligible                                                                                                            | negligible                      | >0                                                                                                                   |
| DLRMv3                   | ```used_bw = tput x (2 + 6*uih_seq_len + 6*num_candidates) x int64_size```                                            | ```used_bw = tput x 98368```    | ```uih_seq_len = 1; num_candidates = 2048; int64_size = 8B```                                                        |

### Egress bandwidth requirements

According to the [rules set out by MLCommons](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#b2-egress-bandwidth), we only need to measure egress bandwidth for benchmarks with large outputs.

| Benchmark                | Formula                                                                                                               | Bandwidth used (bytes)           | Values                                                                                                               |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------------|
| DeepSeek-R1              | ```used_bw = tput x max_output_len x dtype_size```                                                                    | ```used_bw = tput x 80000```     | ```max_output_len = 20000; dtype_size = 4B```                                                                        |
| GPT-OSS-120B             | ```used_bw = tput x max_output_len x dtype_size```                                                                    | ```used_bw = tput x 131072```    | ```max_output_len = 32768; dtype_size = 4B```                                                                        |
| WAN-2.2-T2V-A14B         | ```used_bw = tput x num_frames x height x width x channels x dtype_size```                                            | ```used_bw = tput x 223948800``` | ```num_frames = 81; height = 720; width = 1280; channels = 3; dtype_size = 1B```                                     |
| Qwen3-VL-235B-A22B       | ```used_bw = tput x json_response_size x dtype_size```                                                                | ```used_bw = tput x 2000```      | ```json_response_size = 500; dtype_size = 4B```                                                                      |


## Network bandwidth of NVIDIA's systems

### B200-SXM-180GBx8 (DGX B200) and B300-SXM-288GBx8 (DGX B300)
The [DGX B200/B300 User guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) specifies the network card description as below.
2 x NVIDIA® BlueField®-3 DPU Dual Port Cards. Each card provides the following speeds:
- Ethernet (1 port): 400GbE, 200GbE, 100GbE, 50GbE, 40GbE, 25GbE, and 10GbE
- InfiniBand (1 port): Up to 400Gbps

Thus, each BlueField-3 DPU card allows for at least 400Gbps via InfiniBand and 400GbE via ethernet, amounting to 800Gbps. Since each system has 2 BlueField-3 DPU cards, the total bandwidth is at least 2 x 800Gbps = 1600Gbps = 200GB/s.

### GB200-NVL72_GB200-186GB_aarch64x72 (GB200 NVL72) and GB300-NVL72_GB300-288GB_aarch64x72 (GB300 NVL72)
GB200/GB300 NVL72 has 18 compute nodes. Each compute node has 2 Grace CPUs.
Each Grace CPU is connected to at least one BlueField-3 Smart NIC offering up to 400Gbps = 50GB/s. Thus, each node offers at least 100GB/s network bandwidth. With 18 nodes, GB200/GB300 NVL72 provides at least 1800GB/s aggregate network bandwidth.


## Max permissible QPS per system
Using the formulae in the previous section, we calculate for each system-benchmark pair the maximum permissible QPS by setting `system_bw = used_bw` and calculating for `tput`.
For workloads with constraint on both ingress and egress bandwidth, we take the max of the two (e.g., we take egress for WAN-2.2-T2V-A14B and GPT-OSS-120B).

PLEASE NOTE - The numbers are calculated below for NVIDIA's systems and are provided for reference only. Each system's configuration (and hence, bandwidth) may be different. It is imperative that each participant does such calculations individually for their own systems.

| System                                | Bandwidth (bytes/sec)         | Llama3.1-8B  | Llama2-70B   | Llama3.1-405B | DeepSeek-R1  | GPT-OSS-120B | WAN-2.2-T2V-A14B | Qwen3-VL-235B-A22B | Whisper      | R-GAT       | DLRMv3       |
|---------------------------------------|-------------------------------|--------------|--------------|---------------|--------------|--------------|------------------|---------------------|--------------|-------------|--------------|
| B200-SXM-180GBx8                      | 200GB/s (= 2 x 10^11)         | 1.95 x 10^7  | 4.88 x 10^7  | 2.5 x 10^6    | 2.5 x 10^6   | 1.53 x 10^6  | 893              | 1.85 x 10^6         | 2.08 x 10^5  | N/A         | 2.03 x 10^6  |
| B300-SXM-288GBx8                      | 200GB/s (= 2 x 10^11)         | 1.95 x 10^7  | 4.88 x 10^7  | 2.5 x 10^6    | 2.5 x 10^6   | 1.53 x 10^6  | 893              | 1.85 x 10^6         | 2.08 x 10^5  | N/A         | 2.03 x 10^6  |
| GB200-NVL72_GB200-186GB_aarch64x72    | 1800GB/s (= 1.8 x 10^12)      | 1.76 x 10^8  | 4.39 x 10^8  | 2.25 x 10^7   | 2.25 x 10^7  | 1.37 x 10^7  | 8,039            | 1.66 x 10^7         | 1.875 x 10^6 | N/A         | 1.83 x 10^7  |
| GB300-NVL72_GB300-288GB_aarch64x72    | 1800GB/s (= 1.8 x 10^12)      | 1.76 x 10^8  | 4.39 x 10^8  | 2.25 x 10^7   | 2.25 x 10^7  | 1.37 x 10^7  | 8,039            | 1.66 x 10^7         | 1.875 x 10^6 | N/A         | 1.83 x 10^7  |