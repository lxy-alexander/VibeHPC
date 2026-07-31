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
from code.common.constants import AliasedName, AliasedNameEnum


class SDXLComponent(AliasedNameEnum):
    """Names of supported components for SDXL."""

    CLIP1: AliasedName = AliasedName("clip1", ("clip",))
    CLIP2: AliasedName = AliasedName("clip2", ("clipwithproj", "clip-with-proj",))
    UNETXL: AliasedName = AliasedName("unet", ("unetxl", "unet-base", "unetxl-base",))
    VAE: AliasedName = AliasedName("vae", ("vae-dec", "vae_dec",))
