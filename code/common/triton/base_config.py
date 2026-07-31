# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
G_TRITON_BASE_CONFIG = """
name: "{model_name}"
backend: "tensorrtllm"
max_batch_size: 1

model_transaction_policy {{
  decoupled: {is_decoupled}
}}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
    allow_ragged_batch: true
  }},
  {{
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: {{ shape: [ ] }}
  }},
  {{
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "end_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: {{ shape: [ ] }}
    optional: true
  }},
  {{
    name: "pad_id"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: {{ shape: [ ] }}
    optional: true
  }},
  {{
    name: "beam_width"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: {{ shape: [ ] }}
    optional: true
  }},
  {{
    name: "streaming"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "stop_words_list"
    data_type: TYPE_INT32
    dims: [ 1, 2, -1 ]
    optional: true
  }}
]
output [
  {{
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1, -1 ]
  }},
  {{
    name: "sequence_length"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]
instance_group [
  {{
    count: 1
    kind : KIND_CPU
  }}
]
parameters: {{
  key: "max_beam_width"
  value: {{
    string_value: "{beam_width}"
  }}
}}
parameters: {{
  key: "FORCE_CPU_ONLY_INPUT_TENSORS"
  value: {{
    string_value: "no"
  }}
}}
parameters: {{
  key: "gpt_model_type"
  value: {{
    string_value: "inflight_fused_batching"
  }}
}}
parameters: {{
  key: "gpt_model_path"
  value: {{
    string_value: "{engine_path}"
  }}
}}
parameters: {{
  key: "batch_scheduler_policy"
  value: {{
    string_value: "max_utilization"
  }}
}}
parameters: {{
  key: "kv_cache_free_gpu_mem_fraction"
  value: {{
    string_value: "0.9"
  }}
}}
parameters: {{
  key: "exclude_input_in_output"
  value: {{
    string_value: "true"
  }}
}}
parameters: {{
  key: "enable_kv_cache_reuse"
  value: {{
    string_value: "false"
  }}
}}
parameters: {{
  key: "enable_chunked_context"
  value: {{
    string_value: "{enable_chunked_context}"
  }}
}}
parameters: {{
  key: "gpu_device_ids"
  value: {{
    string_value: "{gpu_device_idx}"
  }}
}}
parameters: {{
  key: "max_num_tokens"
  value: {{
    string_value: "{max_num_tokens}"
  }}
}}
parameters: {{
  key: "guided_decoding_backend"
  value: {{
    string_value: "${{guided_decoding_backend}}"
  }}
}}
parameters: {{
  key: "tokenizer_dir"
  value: {{
    string_value: "${{tokenizer_dir}}"
  }}
}}
parameters: {{
  key: "xgrammar_tokenizer_info_path"
  value: {{
    string_value: "${{xgrammar_tokenizer_info_path}}"
  }}
}}
"""
