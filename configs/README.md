# Config Naming Conventions

## Config Types

There are two kinds of configs:

1. **trtllm serve configs** - 3 types (names are convention, not enforced):
   - `ifb_config` - InFlight Batching
   - `ctx_config` - Context Server (Disaggregated Serving)
   - `gen_config` - Generation Server (Disaggregated Serving)

2. **harness configs** - `harness_config`
   - Only requires harness-specific fields

## Directory Selection

The `SYSTEM_NAME` value in make commands determines which config directory is used.

## Guidelines

- **TPS** and **query count** should match the number of GPUs in the system config.
