"""
This script is supposed to run with llm-compressor, which is not setup in this repository.
Please do `pip install llm-compressor` and other dependencies at your own environment.
"""

from io import BytesIO
import base64
import json
from datetime import timedelta

import torch
from datasets import load_dataset
from openai.types import ResponseFormatJSONSchema
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, field_validator
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# NOTE: Requires a minimum of transformers 4.57.0
class ProductMetadata(BaseModel):
    """Expected JSON schema for the VLM response."""

    category: str
    """The complete category of the product, e.g., "Clothing & Accessories > Clothing > Shirts > Polo Shirts"."""

    brands: list[str]
    """The brands of the product, e.g., ["giorgio armani", "hugo boss"]."""

    is_secondhand: bool
    """True if the product is second-hand, False otherwise."""

class BaseModelWithAttributeDescriptionsFromDocstrings(BaseModel):
    """Base model that automatically adds attribute descriptions from docstrings."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Pydantic settings for
    - Automatically add the attribute descriptions from docstrings.
    - Forbid extra attributes.
    """


_DEFAULT_DATASET_SIZE = 48289
_DEFAULT_MIN_DURATION = timedelta(minutes=10)
_DEFAULT_OFFLINE_EXPECTED_QPS = (
    _DEFAULT_DATASET_SIZE / _DEFAULT_MIN_DURATION.total_seconds()
)

class LoadedSample(BaseModelWithAttributeDescriptionsFromDocstrings):
    """Sample format to be used by LoadGen."""

    messages: list[ChatCompletionMessageParam]
    """The messages to be sent for chat completion to the VLM inference endpoint."""

    response_format: ResponseFormatJSONSchema | None = None
    """The response format to be used during guided decoding."""

    @field_validator("messages", mode="after")
    @classmethod
    def ensure_content_is_list(
        cls,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        """If the content is a `ValidatorIterator`, convert it back to a list.

        This is to workaround a Pydantic bug. See
        https://github.com/pydantic/pydantic/issues/9467 for more details.
        """
        for message in messages:
            if (
                "content" in message
                and message["content"].__class__.__module__
                == "pydantic_core._pydantic_core"
                and message["content"].__class__.__name__ == "ValidatorIterator"
            ):
                message["content"] = list(
                    message["content"])  # type: ignore[arg-type]
        return messages

def load_shopify_dataset(
    repo_id: str = "Shopify/the-catalogue-public-beta",
    splits: list[str] = ["train", "test"],
    token: str | None = None,
):
    """Load the Shopify dataset from HuggingFace.

    Args:
        repo_id: The HuggingFace repository ID of the dataset.
        splits: Dataset splits to load.
        token: Optional HuggingFace token for private datasets.

    Returns:
        The loaded dataset.
    """
    return load_dataset(
        repo_id,
        token=token,
        split="+".join(splits),
        revision="main",
    )

def process_sample_to_vllm_messages(
    sample: dict,
    use_guided_decoding: bool = False,
) -> LoadedSample:
    """Formulate the sample to be loaded into host memory before testing.

    Args:
        sample: The sample from the dataset to be formulated into a loaded sample.
        use_guided_decoding: Whether to use guided decoding for the sample.

    Returns:
        The loaded sample to be used for issuing queries to the inference endpoint.
    """
    image_file = BytesIO()
    image_format = sample["product_image"].format
    sample["product_image"].save(image_file, format=image_format)
    image_bytes = image_file.getvalue()
    image_base64 = base64.b64encode(image_bytes)
    image_base64_string = image_base64.decode("utf-8")
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please analyze the product from the user prompt
and provide the following fields in a valid JSON object:
- category
- brand
- is_secondhand

You must choose only one, which is the most appropriate, correct, and specifc
category out of the list of possible product categories.

The description of the product sometimes contains various types of source code
(e.g., JavaScript, CSS, HTML, etc.), where useful product information is embedded
somewhere inside the source code. For this task, you should extract the useful
product information from the source code and leverage it, and discard the
programmatic parts of the source code.

Your response should only contain a valid JSON object and nothing more, e.g.,
you should not fence the JSON object inside a ```json code block.
The JSON object should match the followng JSON schema:
```json
{json.dumps(ProductMetadata.model_json_schema(), indent=2)}
```
""",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""The title of the product is the following:
```text
{sample['product_title']}
```

The description of the product is the following:
```text
{sample['product_description']}
```

The following are the possible product categories:
```json
{sample['potential_product_categories']}
```
""",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,"
                        f"{image_base64_string}",
                    },
                },
            ],
        },
    ]

    return LoadedSample(
        messages=messages,
        response_format=(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "product_metadata",
                    "schema": ProductMetadata.model_json_schema(),
                    "strict": True,
                },
            }
            if use_guided_decoding
            else None
        ),
    )





MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 65536
ds = load_shopify_dataset()

# Load model
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)


def preprocess_function(example):
    messages = process_sample_to_vllm_messages(example).messages

    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)

# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    dataset=ds,
    data_collator=data_collator,
)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")


#Save to disk in compressed-tensors format.
SAVE_DIR = "/opt" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
