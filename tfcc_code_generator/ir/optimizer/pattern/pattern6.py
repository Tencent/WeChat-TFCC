# Copyright 2021 Wechat Group, Tencent
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

import numpy as np
import ir.framework
import ir.node
from ir.optimizer.patternmanager import NodeManager
from .pattern import Pattern


class Pattern6(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 3 or len(node_map) == 4

        if rule_name == "SRC0":
            get_shape_node = node_map["s0_get_shape"]
            slice_node = node_map["s0_slicev2"]
        else:
            get_shape_node = node_map["s1_get_shape"]
            slice_node = node_map["s1_slicev2"]

        if not slice_node.inputs[1].is_constant():
            return False
        if not slice_node.inputs[2].is_constant():
            return False

        start = slice_node.inputs[1].data.tolist()[0]
        end = slice_node.inputs[2].data.tolist()[0]

        value = get_shape_node.inputs[0].shape[start:end]
        if not all([isinstance(s, int) for s in value]):
            return False

        output_name = slice_node.output_names[0]
        dtype = slice_node.outputs[0].dtype

        # unref
        node_manager.unrefer_node_output(slice_node, 0)

        symbol = ir.framework.Symbol(output_name)
        symbol.stype = ir.framework.SymbolType.CONSTANT_TENSOR
        symbol.origin_stype = symbol.stype
        symbol.shape = [len(value)]
        symbol.dtype = dtype
        symbol.data = np.asarray(value, dtype=dtype.numpy_dtype)
        assert symbol.verify()
        node_manager.graph.add_symbol(symbol)
        node_manager.graph.add_keep(symbol.name)

        return True
