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

from ir.node import Node
import ir.framework


class If(Node):
    def update_attributes(
        self,
        then_graph_name: str,
        then_graph_capture_count: int,
        else_graph_name: str,
        else_graph_capture_count: int,
    ):
        assert (
            isinstance(then_graph_name, str)
            and then_graph_name in self.graph.model.graphs
        )
        assert (
            isinstance(then_graph_capture_count, int) and then_graph_capture_count >= 0
        )
        assert (
            isinstance(else_graph_name, str)
            and else_graph_name in self.graph.model.graphs
        )
        assert (
            isinstance(else_graph_capture_count, int) and else_graph_capture_count >= 0
        )
        self._then_graph_name = then_graph_name
        self._then_graph_capture_count = then_graph_capture_count
        self._else_graph_name = else_graph_name
        self._else_graph_capture_count = else_graph_capture_count

    # inputs: [cond]
    def inference(self):
        assert (
            len(self.input_names)
            == 1 + self.then_graph_capture_count + self.else_graph_capture_count
        )

        then_graph = self.model.graphs[self.then_graph_name]
        else_graph = self.model.graphs[self.else_graph_name]

        assert len(then_graph.outputs) == len(self.output_names)
        assert len(else_graph.outputs) == len(self.output_names)

        for i, (then_graph_output_name, else_graph_output_name) in enumerate(
            zip(then_graph.outputs, else_graph.outputs)
        ):
            then_graph_output_symbol = then_graph.get_symbol(then_graph_output_name)
            else_graph_output_symbol = else_graph.get_symbol(else_graph_output_name)
            assert then_graph_output_symbol.dtype == else_graph_output_symbol.dtype
            if then_graph_output_symbol.is_value():
                assert else_graph_output_symbol.is_value()
            if then_graph_output_symbol.is_tensor():
                assert else_graph_output_symbol.is_tensor()
            if then_graph_output_symbol.is_vector():
                assert else_graph_output_symbol.is_vector()
            assert len(then_graph_output_symbol.shape) == len(
                else_graph_output_symbol.shape
            )
            shape = []
            for s1, s2 in zip(
                then_graph_output_symbol.shape, else_graph_output_symbol.shape
            ):
                if s1 == s2:
                    shape.append(s1)
                else:
                    shape.append(self.create_shape_name("if"))
            self.outputs[i].dtype = then_graph_output_symbol.dtype
            if then_graph_output_symbol.is_value():
                self.outputs[i].stype = ir.framework.SymbolType.VALUE
            elif then_graph_output_symbol.is_vector():
                self.outputs[i].stype = ir.framework.SymbolType.VECTOR
            elif then_graph_output_symbol.is_tensor():
                self.outputs[i].stype = ir.framework.SymbolType.VARIABLE
            else:
                raise RuntimeError("Unknow stype")
            self.outputs[i].shape = shape

    @property
    def then_graph_name(self):
        return self._then_graph_name

    @property
    def then_graph_capture_count(self):
        return self._then_graph_capture_count

    @property
    def else_graph_name(self):
        return self._else_graph_name

    @property
    def else_graph_capture_count(self):
        return self._else_graph_capture_count

    @property
    def attributes(self):
        return {
            "then_graph_name": self.then_graph_name,
            "then_graph_capture_count": self.then_graph_capture_count,
            "else_graph_name": self.else_graph_name,
            "else_graph_capture_count": self.else_graph_capture_count,
        }
