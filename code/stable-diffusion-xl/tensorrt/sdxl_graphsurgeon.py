#! /usr/bin/env python3

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


import os
import re
import tempfile
import argparse
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import random

from pathlib import Path
from code.common import logging, run_command
from polygraphy.backend.onnx.loader import fold_constants
from nvmitten.constants import Precision
from nvmitten.nvidia.builder import ONNXNetwork

from code.common import logging


__doc__ = """Scripts for modifying SDXL onnx graphs
"""


@gs.Graph.register()
def insert_cast(self, input_tensor, attrs):
    """
    Create a cast layer using tensor as input.
    """
    output_tensor = gs.Variable(name=f'{input_tensor.name}/Cast_output',
                                dtype=attrs['to'])
    next_node_list = input_tensor.outputs.copy()
    self.layer(op='Cast',
               name=f'{input_tensor.name}/Cast',
               inputs=[input_tensor],
               outputs=[output_tensor],
               attrs=attrs)

    # use cast output as input to next node
    for next_node in next_node_list:
        for idx, next_input in enumerate(next_node.inputs):
            if next_input.name == input_tensor.name:
                next_node.inputs[idx] = output_tensor


class SDXLGraphSurgeon(ONNXNetwork):
    """
    The class is the base class to optimize onnx models converted from SDXL pytorch models.
    """

    # onnx threshold of using onnx.save_model instead of onnx.save
    ONNX_LARGE_FILE_THRESHOLD = 2 ** 31

    def __init__(self,
                 onnx_path,
                 precision,
                 device_type,
                 model_name,
                 add_hidden_states=False):
        super().__init__(onnx_path,
                         Precision.FP16,  # TODO yihengz: Overwrite SDXL precision to bypass calibration cache load because we use explicit quantized model, update after picking up mitten fix
                         op_name_remap=dict())  # No rename for SDXL
        self.device_type = device_type
        self.name = model_name
        self.add_hidden_states = add_hidden_states
        self.fp8_unet = False
        self.int8_unet = False
        self.fp16_vae = False
        logging.info(f"{self.name} {precision}: add_hidden_states = {self.add_hidden_states}")
        if model_name == "unet":
            if precision == Precision.FP8:
                self.fp8_unet = True
                logging.info(f"Unet GS path: fp8")
            elif precision == Precision.INT8:
                self.int8_unet = True
                logging.info(f"Unet GS path: int8")
        elif model_name == "vae" and precision == Precision.INT8:
            self.fp16_vae = True

    def info(self, prefix):
        logging.info(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=False)
        self.graph = gs.import_onnx(onnx_graph)

    def infer_shapes(self):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > SDXLGraphSurgeon.ONNX_LARGE_FILE_THRESHOLD:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                tmp_path.mkdir(exist_ok=True)
                onnx_orig_path = tmp_path / "model.onnx"
                onnx_inferred_path = tmp_path / "inferred.onnx"
                onnx.save_model(onnx_graph,
                                str(onnx_orig_path),
                                save_as_external_data=True,
                                all_tensors_to_one_file=True,
                                convert_attribute=False)
                onnx.shape_inference.infer_shapes_path(str(onnx_orig_path), str(onnx_inferred_path))
                onnx_graph = onnx.load(str(onnx_inferred_path))
        else:
            onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
        self.graph = gs.import_onnx(onnx_graph)

    def convert_fp16_io(self):
        for input_tensor in self.graph.inputs:
            input_tensor.dtype = onnx.TensorProto.FLOAT16

        for output_tensor in self.graph.outputs:
            output_tensor.dtype = onnx.TensorProto.FLOAT16

    def convert_fp8_qdq(self):
        onnx_graph = gs.export_onnx(self.graph)

        QDQ_zero_nodes = set()
        # Find all scale and zero constant nodes
        for node in onnx_graph.graph.node:
            if node.op_type == "QuantizeLinear":
                if len(node.input) > 2:
                    QDQ_zero_nodes.add(node.input[2])

        logging.info(f"Found {len(QDQ_zero_nodes)} QDQ pairs")

        # Convert zero point datatype from int8 to fp8
        for node in onnx_graph.graph.node:
            if node.output[0] in QDQ_zero_nodes:
                node.attribute[0].t.data_type = onnx.TensorProto.FLOAT8E4M3FN
        self.graph = gs.import_onnx(onnx_graph)

    def polygraphy_convert_fp16(self):
        onnx_graph = gs.export_onnx(self.graph)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tmp_path.mkdir(exist_ok=True)
            onnx_orig_path = tmp_path / "model.onnx"
            onnx.save_model(onnx_graph,
                            str(onnx_orig_path),
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            convert_attribute=False)
            polygraphy_fp16_path = tmp_path / "fp6.ply.onnx"
            run_command(f"polygraphy convert --fp-to-fp16 -o {str(polygraphy_fp16_path)} {str(onnx_orig_path)}")
            self.graph = gs.import_onnx(onnx.load(str(polygraphy_fp16_path)))

    def remove_resblock_fp32_cast(self):
        """
        Remove unwanted Cast (fp16 -> fp32)
        """
        nodes = self.graph.nodes

        node_fp32_cast_regex = '\/.+\/resnets.\d+\/Cast.*'
        node_fp32_cast = [_n for _n in nodes if re.match(node_fp32_cast_regex, _n.name)]

        logging.info(f"Found {len(node_fp32_cast)} unwanted Cast nodes")

        for cast_node in node_fp32_cast:
            assert cast_node.op == 'Cast'
            next_node = cast_node.o()
            input_tensor = cast_node.inputs[0]
            output_tensor = cast_node.outputs[0]

            target_idx = None
            for idx, next_input_tensor in enumerate(next_node.inputs):
                if next_input_tensor.name == output_tensor.name:
                    target_idx = idx
            next_node.inputs[target_idx] = input_tensor

            cast_node.inputs = []
            cast_node.outputs = []

    def update_resize(self):
        nodes = self.graph.nodes
        up_block_resize_regex = '\/up_blocks.0\/upsamplers.0\/Resize|\/up_blocks.1\/upsamplers.0\/Resize'
        up_block_resize_nodes = [_n for _n in nodes if re.match(up_block_resize_regex, _n.name)]

        logging.info(f"Found {len(up_block_resize_nodes)} Resize nodes to fix")
        for resize_node in up_block_resize_nodes:
            for input_tensor in resize_node.inputs:
                if input_tensor.name:
                    self.graph.insert_cast(input_tensor=input_tensor, attrs={'to': np.float32})
            for output_tensor in resize_node.outputs:
                if output_tensor.name:
                    self.graph.insert_cast(input_tensor=output_tensor, attrs={'to': np.float16})

    def insert_fp8_mha_cast(self):
        def remove_dummy_add_ask(add_node):
            assert add_node.op == 'Add'
            add_node.o().inputs = add_node.i().outputs  # set BMM1 scale output as Softmax input
            add_node.inputs = []

        nodes = self.graph.nodes
        tensors = self.graph.tensors()
        tensor_names = tensors.keys()

        node_dummy_add_mask_regex = '\/.+\/attentions.\d+\/transformer_blocks.\d+\/attn\d+\/Add'
        tensor_qkv_dq_output_regex = '\/.+\/attentions.\d+\/transformer_blocks.\d+\/attn\d+\/[qkv]_bmm_quantizer\/DequantizeLinear_output_0'
        tensor_softmax_scale_input_regex = '\/.+\/attentions.\d+\/transformer_blocks.\d+\/attn\d+\/MatMul_output_0'
        tensor_softmax_dq_output_regex = '\/.+\/attentions.\d+\/transformer_blocks.\d+\/attn\d+\/softmax_quantizer\/DequantizeLinear_output_0'
        tensor_bmm2_output_regex = '\/.+\/attentions.\d+\/transformer_blocks.\d+\/attn\d+\/MatMul_1_output_0'

        node_dummy_add_masks = [_n for _n in nodes if re.match(node_dummy_add_mask_regex, _n.name)]
        tensor_qkv_dq_outputs = [tensors[tensor_name] for tensor_name in tensor_names if re.match(tensor_qkv_dq_output_regex, tensor_name)]
        tensor_softmax_scale_inputs = [tensors[tensor_name] for tensor_name in tensor_names if re.match(tensor_softmax_scale_input_regex, tensor_name)]
        tensor_softmax_dq_outputs = [tensors[tensor_name] for tensor_name in tensor_names if re.match(tensor_softmax_dq_output_regex, tensor_name)]
        tensor_bmm2_outputs = [tensors[tensor_name] for tensor_name in tensor_names if re.match(tensor_bmm2_output_regex, tensor_name)]

        logging.info(f"Found {len(node_dummy_add_masks)} attentions")

        # remove dummy add mask
        for node in node_dummy_add_masks:
            remove_dummy_add_ask(node)

        # TRT 10.2 fp8 MHA required onnx pattern
        #   Q           K           V
        #   |           |           |
        #   to_fp32   to_fp32     to_fp32
        #   \          /           |
        #      BMM1                |
        #       |                  |
        #     to_fp16             /
        #       |               /
        #     scale           /
        #       |           /
        #     SoftMax     /
        #       |       /
        #     to_fp32 /
        #       |   /
        #      BMM2
        #       |
        #     to_fp16
        #       |

        for tensor in tensor_qkv_dq_outputs:
            self.graph.insert_cast(input_tensor=tensor, attrs={'to': np.float32})

        for tensor in tensor_softmax_scale_inputs:
            self.graph.insert_cast(input_tensor=tensor, attrs={'to': np.float16})

        for tensor in tensor_softmax_dq_outputs:
            self.graph.insert_cast(input_tensor=tensor, attrs={'to': np.float32})

        for tensor in tensor_bmm2_outputs:
            self.graph.insert_cast(input_tensor=tensor, attrs={'to': np.float16})

    def clip_add_hidden_states(self):

        nodes = self.graph.nodes
        tensors = self.graph.tensors()
        tensor_names = tensors.keys()

        gather_pat = r"\/text_model\/embeddings\/token_embedding\/Gather*"

        # collect gather_nodes
        gather_nodes = [_n for _n in nodes if re.match(gather_pat, _n.name)]
        for node in gather_nodes:
            self.graph.insert_cast(input_tensor=node.inputs[1], attrs={'to': np.int32})  # weight cast
        logging.info(f"Found {len(gather_nodes)} gather's weight to convert to int32")


        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers - 1):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"

            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers - 1):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        self.graph = gs.import_onnx(onnx_graph)

    def fuse_mha_qkv_int8_sq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()

        # mha  : fuse QKV QDQ nodes
        # mhca : fuse KV QDQ nodes
        q_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/to_q/input_quantizer/DequantizeLinear_output_0'
        k_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/to_k/input_quantizer/DequantizeLinear_output_0'
        v_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/to_v/input_quantizer/DequantizeLinear_output_0'

        qs = list(sorted(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(q_pat, key) for key in keys]))))
        ks = list(sorted(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(k_pat, key) for key in keys]))))
        vs = list(sorted(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(v_pat, key) for key in keys]))))

        removed = 0
        assert len(qs) == len(ks) == len(vs), 'Failed to collect tensors'
        for q, k, v in zip(qs, ks, vs):
            is_mha = all(['attn1' in tensor for tensor in [q, k, v]])
            is_mhca = all(['attn2' in tensor for tensor in [q, k, v]])
            assert (is_mha or is_mhca) and (not (is_mha and is_mhca))

            if is_mha:
                tensors[k].outputs[0].inputs[0] = tensors[q]
                tensors[v].outputs[0].inputs[0] = tensors[q]
                del tensors[k]
                del tensors[v]

                removed += 2

            else:  # is_mhca
                tensors[k].outputs[0].inputs[0] = tensors[v]
                del tensors[k]

                removed += 1

        return removed  # expected 72 for L2.5

    def remove_FC_int8_qdq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()
        nodes = {node.name: node for node in self.graph.nodes}

        # remove QDQ nodes from linear layers after MHA/MHCA
        A_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/Reshape_7_output_0'
        B_pat = 'down_blocks.\d+.attentions.\d+.transformer_blocks.\d+.attn\d+.to_out.0.weight'
        target_pat = '/down_blocks.\d+/attentions.\d+/transformer_blocks.\d+/attn\d+/to_out.0/MatMul'

        As = list(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(A_pat, key) for key in keys])))
        Bs = list(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(B_pat, key) for key in keys])))
        targets = list(map(lambda x: x.group(0), filter(lambda x: x is not None, [re.match(target_pat, key) for key in keys])))

        removed = 0
        for A, B, target in zip(As, Bs, targets):
            target_node = nodes[target]

            A_, B_ = target_node.inputs
            del tensors[A_.name]
            del tensors[B_.name]

            removed += 2
            target_node.inputs = [tensors[A], tensors[B]]

        return removed  # expected 96 for L2.5

    def insert_vae_fp16_conv_cast(self):
        """
        Insert fp32 -> fp16 cast for VAE input / mid_block / up_blocks convs
        """

        nodes = self.graph.nodes
        tensors = self.graph.tensors()
        tensor_names = tensors.keys()

        # regex patterns for VAE input/mid_block/up_blocks conv nodes & input/output tensors
        conv_pat = r"\/decoder\/((((up_blocks.[23]\/resnets.[12])|(mid_block\/resnets.[01]))\/conv[12])|conv_in)\/Conv"
        conv_input_pat = r"\/((decoder\/((up_blocks.[23]\/resnets.[12])|(mid_block\/resnets.[01]))\/nonlinearity(_1)?\/Mul_output_0)|post_quant_conv\/Conv_output_0)"
        conv_output_pat = r"\/decoder\/((((up_blocks.[23]\/resnets.[12])|(mid_block\/resnets.[01]))\/conv[12])|conv_in)\/Conv_output_0"

        # collect input/mid_block/up_blocks conv nodes/tensors
        conv_nodes = [_n for _n in nodes if re.match(conv_pat, _n.name)]
        conv_input_tensors = [tensors[tensor_name] for tensor_name in tensor_names if re.match(conv_input_pat, tensor_name)]
        conv_output_tensors = [tensors[tensor_name] for tensor_name in tensor_names if re.match(conv_output_pat, tensor_name)]

        logging.info(f"Found {len(conv_nodes)} input/midblock/upblock convs to convert to fp16")

        # insert input tensor cast to fp16
        for tensor in conv_input_tensors:
            self.graph.insert_cast(input_tensor=tensor, attrs={'to': np.float16})

        # insert output tensor cast to fp32
        for tensor in conv_output_tensors:
            self.graph.insert_cast(input_tensor=tensor, attrs={'to': np.float32})

        # cast weight/bias node inputs to float16 numpy array
        for node in conv_nodes:
            self.graph.insert_cast(input_tensor=node.inputs[1], attrs={'to': np.float16})  # weight cast
            self.graph.insert_cast(input_tensor=node.inputs[2], attrs={'to': np.float16})  # bias cast

    def insert_vae_fp16_linear_casts(self):
        """
        Insert fp32 -> fp16 casts for the following linear layers:
            - mid_block/attentions.0 to_[kqv] & to_out.0 Mul -> MatMul -> Adds
            - additional 2 mid_block/attentions.0 MatMuls
            - up_block & conv_norm_out Mul -> Adds
        """

        nodes = self.graph.nodes
        tensors = self.graph.tensors()
        tensor_names = tensors.keys()

        # mid_block/attentions.0 to_[kqv] & to_out.0 Mul -> MatMul -> Adds
        mul_kqv_pat = r"\/decoder\/mid_block\/attentions.0\/(to_[kqv]|to_out.0)\/input_quantizer\/Mul"
        matmul_kqv_pat = r"\/decoder\/mid_block\/attentions.0\/(to_[kqv]|to_out.0)\/MatMul"
        add_kqv_pat = r"\/decoder\/mid_block\/attentions.0\/(to_[kqv]|to_out.0)\/Add"
        mul_kqv_input_pat = r"\/decoder\/mid_block\/attentions.0\/(Transpose_1_output_0|Cast_4_output_0)"
        add_kqv_output_pat = r"\/decoder\/mid_block\/attentions.0\/(to_[kqv]|to_out.0)\/Add_output_0"

        mul_kqv_nodes = [_n for _n in nodes if re.match(mul_kqv_pat, _n.name)]
        matmul_kqv_nodes = [_n for _n in nodes if re.match(matmul_kqv_pat, _n.name)]
        add_kqv_nodes = [_n for _n in nodes if re.match(add_kqv_pat, _n.name)]
        mul_kqv_input_tensors = [tensors[tensor_name] for tensor_name in tensor_names if re.match(mul_kqv_input_pat, tensor_name)]
        add_kqv_output_tensors = [tensors[tensor_name] for tensor_name in tensor_names if re.match(add_kqv_output_pat, tensor_name)]

        # additional 2 mid_block/attentions.0 MatMuls
        matmul_pat = r"\/decoder\/mid_block\/attentions.0\/MatMul(_1)?"
        matmul_input_pat = r"\/decoder\/mid_block\/attentions.0\/(Mul_[12]|Softmax|Transpose_3)_output_0"
        matmul_output_pat = r"\/decoder\/mid_block\/attentions.0\/MatMul_(1_)?output_0"

        matmul_nodes = [_n for _n in nodes if re.match(matmul_pat, _n.name)]
        matmul_input_tensors = [tensors[tensor_name] for tensor_name in tensor_names if re.match(matmul_input_pat, tensor_name)]
        matmul_output_tensors = [tensors[tensor_name] for tensor_name in tensor_names if re.match(matmul_output_pat, tensor_name)]

        # up_block & conv_norm_out Mul -> Adds
        mul_pat = r"\/decoder\/((up_blocks.[0123]\/resnets.[012]\/norm[12])|conv_norm_out)\/Mul"
        add_pat = r"\/decoder\/((up_blocks.[0123]\/resnets.[012]\/norm[12])|conv_norm_out)\/Add"
        mul_input_pat = r"\/decoder\/((up_blocks.[0123]\/resnets.[012]\/norm[12])|conv_norm_out)\/Reshape_1_output_0"
        add_output_pat = r"\/decoder\/((up_blocks.[0123]\/resnets.[012]\/norm[12])|conv_norm_out)\/Add_output_0"

        mul_nodes = [_n for _n in nodes if re.match(mul_pat, _n.name)]
        add_nodes = [_n for _n in nodes if re.match(add_pat, _n.name)]
        mul_input_tensors = [tensors[tensor_name] for tensor_name in tensor_names if re.match(mul_input_pat, tensor_name)]
        add_output_tensors = [tensors[tensor_name] for tensor_name in tensor_names if re.match(add_output_pat, tensor_name)]

        # collect input/output tensors for casting
        input_tensors = mul_kqv_input_tensors + matmul_input_tensors + mul_input_tensors
        output_tensors = add_kqv_output_tensors + matmul_output_tensors + add_output_tensors

        # insert input tensor cast to fp16
        for tensor in input_tensors:
            self.graph.insert_cast(input_tensor=tensor, attrs={'to': np.float16})

        # insert output tensor cast to fp32
        for tensor in output_tensors:
            self.graph.insert_cast(input_tensor=tensor, attrs={'to': np.float32})

        # muls / matmuls / adds in Mul -> Add need 1st index input cast to FP16
        for node in (mul_kqv_nodes + matmul_kqv_nodes + matmul_nodes + mul_nodes + add_nodes):
            self.graph.insert_cast(input_tensor=node.inputs[1], attrs={'to': np.float16})

        # adds in Mul -> MatMul -> Add need 0th input cast to FP16
        for node in add_kqv_nodes:
            self.graph.insert_cast(input_tensor=node.inputs[0], attrs={'to': np.float16})

    def prefusion(self):
        """
        Append the Non-Maximum Suppression (NMS) layer to the conv heads
        """
        self.info(f'{self.name}: original')

        if self.int8_unet:
            if (removed := self.fuse_mha_qkv_int8_sq()) > 0:
                self.info(f'{self.name}: removing {removed} mha qkv int8 sq nodes')

            if (removed := self.remove_FC_int8_qdq()) > 0:
                self.info(f'{self.name}: removing {removed} qdq nodes for FC after mha/mhca')

        # perform VAE fp32 -> fp16 casts for midblock/upblock convs & linear layers
        if self.fp16_vae:
            self.insert_vae_fp16_conv_cast()
            self.info(f'{self.name}: insert fp32 -> fp16 casts for midblock/upblock convs')

            self.insert_vae_fp16_linear_casts()
            self.info(f'{self.name}: insert fp32 -> fp16 casts for linear layers')

        self.cleanup_graph()
        self.info(f'{self.name}: cleanup')

        if self.fp8_unet:
            # QDQ WAR, has to happen before constant folding, WAR it until ModelOpt fixes it
            self.convert_fp8_qdq()
            self.info(f'{self.name}: convert qdq to fp8')

            self.insert_fp8_mha_cast()
            self.info(f'{self.name}: insert cast for fp8 mha')

            # TRT complains about resize in 2 upsamplers, WAR it until ModelOpt fixes it
            self.update_resize()

            # fp8 unet engine needs strongly typed onnx, convert io tensor from fp32 to fp16
            self.convert_fp16_io()
        else:
            self.fold_constants()
            self.info(f'{self.name}: fold constants')

            self.infer_shapes()
            self.info(f'{self.name}: shape inference')

        if self.add_hidden_states:
            self.clip_add_hidden_states()
            self.info(f'{self.name}: added hidden_states')

        self.info(f'{self.name}: GS finished')


def parse_args():
    """
    Arguments that can be used for standalone run
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx-fpath',
                        type=str,
                        default='build/models/retinanet-resnext50-32x4d/submission/retinanet_resnext50_32x4d_efficientNMS.800x800.onnx',
                        help='Input ONNX file for ResNet50')
    parser.add_argument('--output-onnx-fpath',
                        type=str,
                        default='/tmp/sdxl_graphsurgeon.onnx',
                        help='Output ONNX filename')
    parser.add_argument('--precision',
                        type=str,
                        default='fp16',
                        choices={'fp16', 'fp8', 'int8'},
                        help='Compute precision')
    parser.add_argument('--name',
                        type=str,
                        default='unet',
                        help='Model name')
    parser.add_argument('--hidden_states',
                        action="store_true",
                        help='Add hidden states output for CLIP models')

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            logging.debug("Parsed args -- {}: {}".format(key, value))

    return args


def main(args):
    """
    commandline entrance of the graphsurgeon. Example commands:
        python3 -m code.stable-diffusion-xl.tensorrt.sdxl_graphsurgeon --onnx-fpath=/work/build/sdxl.scratch.4.1/fp8.l4/unet.onnx --precison=fp8 --output-onnx-fpath=/work/build/sdxl.scratch.4.1/mlperf.gs.fp8/unet.onnx --name=unet
    """
    device_type = 'gpu'
    sdxl_gs = SDXLGraphSurgeon(args.onnx_fpath,
                               args.precision,
                               device_type,
                               args.name,
                               args.hidden_states)

    model = sdxl_gs.create_onnx_model()
    os.makedirs(Path(args.output_onnx_fpath).parent, exist_ok=True)
    if model.ByteSize() > SDXLGraphSurgeon.ONNX_LARGE_FILE_THRESHOLD:
        onnx.save_model(model,
                        args.output_onnx_fpath,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        convert_attribute=False)
    else:
        onnx.save(model, args.output_onnx_fpath)


if __name__ == '__main__':
    args = parse_args()
    main(args)
