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

from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node
from ..converter import Converter


class Reduce(Converter):
    def __init__(self, node_cls):
        super().__init__()
        self._node_cls = node_cls

    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        axes = []
        if len(inp_strs) >= 2:
            assert len(inp_strs) == 2
            axes = list(graph.get_symbol(inp_strs[1]).data)
            axes = list(map(int, axes))  # Convert axes[i] from numpy.int32 to int
            for i in range(len(axes)):
                if axes[i] < 0:
                    axes[i] = axes[i] + len(op.inputs[0].shape)

        symbol = graph.get_symbol(inp_strs[0])
        if symbol.is_value():
            if axes:
                return False
            graph.append_node(
                ir.node.base.Identity(op.name, graph, [inp_strs[0]], oup_strs)
            )
            return True

        keep_dims = op.get_attr("keep_dims")
        if keep_dims:
            graph.append_node(
                self._node_cls(op.name, graph, [inp_strs[0]], oup_strs, axes=axes)
            )
        else:
            squeeze_output_name = graph.context.create_symbol_name(oup_strs[0])
            graph.append_node(
                self._node_cls(
                    op.name + ":0",
                    graph,
                    [inp_strs[0]],
                    [squeeze_output_name],
                    axes=axes,
                )
            )

            if list(range(len(symbol.shape))) == axes:
                graph.append_node(
                    ir.node.base.ToValue(
                        op.name + ":1", graph, [squeeze_output_name], oup_strs
                    )
                )
            else:
                graph.append_node(
                    ir.node.base.Squeeze(
                        op.name + ":1",
                        graph,
                        [squeeze_output_name],
                        oup_strs,
                        axes=axes,
                    )
                )
        return True


class Mean(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceMean)


class ReduceMean(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceMean)


class Sum(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceSum)


class ReduceSum(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceSum)


class Prod(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceProd)


class ReduceProd(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceProd)


class Max(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceMax)


class ReduceMax(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceMax)


class Min(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceMin)


class ReduceMin(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceMin)


class All(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceAll)


class ReduceAll(Reduce):
    def __init__(self):
        super().__init__(ir.node.math.ReduceAll)
