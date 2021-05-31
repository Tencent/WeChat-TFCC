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


class Pattern8(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 4 or len(node_map) == 5

        to_tensor_node = node_map[rule_name + "_to_tensor"]
        gather_or_slice_node = node_map[rule_name + "_gather_or_slice"]
        concat_node = node_map[rule_name + "_concat"]

        if not all(
            [symbol.is_constant() for symbol in gather_or_slice_node.inputs[1:]]
        ):
            return False

        for symbol in concat_node.inputs:
            if (
                symbol.name != gather_or_slice_node.output_names[0]
                and not symbol.is_constant()
            ):
                return False
        if len(concat_node.outputs[0].shape) != 1:
            return False

        start = None
        end = None
        if isinstance(
            gather_or_slice_node, (ir.node.base.SliceV1, ir.node.base.SliceV2)
        ):
            if not isinstance(gather_or_slice_node.inputs[0].shape[0], int):
                return False
            start = gather_or_slice_node.inputs[1].data.tolist()[0]
            while start < 0:
                start += gather_or_slice_node.inputs[0].shape[0]
            if isinstance(gather_or_slice_node, ir.node.base.SliceV1):
                end = start + gather_or_slice_node.length
            elif isinstance(gather_or_slice_node, ir.node.base.SliceV2):
                end = gather_or_slice_node.inputs[2].data.tolist()[0]
                while end < 0:
                    end += gather_or_slice_node.inputs[0].shape[0]
            else:
                raise RuntimeError("Unknow error")
            end = min(end, gather_or_slice_node.inputs[0].shape[0])
            if end - start > 4:
                return False

        output_name = concat_node.output_names[0]
        dtype = concat_node.outputs[0].dtype
        node_name_prefix = concat_node.name + "_pattern8"
        output_name_prefix = concat_node.output_names[0] + "_pattern8_"
        insert_index = node_manager.graph.nodes.index(concat_node)

        # unref
        node_manager.unrefer_node_output(concat_node, 0)

        vec = []
        create_vector_input_names = []
        for symbol in concat_node.inputs:
            if symbol.is_constant():
                vec.append(symbol.data.tolist())
            elif symbol.name == gather_or_slice_node.output_names[0]:
                at_node_count = 0
                if isinstance(gather_or_slice_node, ir.node.nn.Gather):
                    ids = gather_or_slice_node.inputs[1].data.tolist()
                else:
                    ids = range(start, end)
                for idx in ids:
                    at_node_output_name = node_manager.graph.context.create_symbol_name(
                        output_name_prefix
                    )
                    at_node = ir.node.base.At1(
                        node_name_prefix + "_at_" + str(at_node_count),
                        node_manager.graph,
                        to_tensor_node.input_names,
                        [at_node_output_name],
                        idx=idx,
                    )
                    at_node_count += 1
                    node_manager.add_node(insert_index, at_node)
                    insert_index += 1
                    vec.append(len(create_vector_input_names))
                    create_vector_input_names.append(at_node_output_name)
            else:
                raise RuntimeError("Unknow error")

        create_vector_node_output_name = node_manager.graph.context.create_symbol_name(
            output_name_prefix
        )
        create_vector_node = ir.node.base.CreateVector(
            node_name_prefix + "_create_vector",
            node_manager.graph,
            create_vector_input_names,
            [create_vector_node_output_name],
            dtype=dtype,
            vec=vec,
        )
        node_manager.add_node(insert_index, create_vector_node)
        insert_index += 1
        new_to_tensor_node = ir.node.base.ToTensor(
            node_name_prefix + "_to_tensor",
            node_manager.graph,
            [create_vector_node_output_name],
            [output_name],
        )
        node_manager.add_node(insert_index, new_to_tensor_node)
        insert_index += 1

        return True
