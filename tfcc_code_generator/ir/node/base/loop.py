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


class Loop(Node):
    def update_attributes(
        self,
        sub_graph_name: str,
        carried_count: int,
        capture_count: int,
        scan_count: int,
    ):
        assert (
            isinstance(sub_graph_name, str)
            and sub_graph_name in self.graph.model.graphs
        )
        assert isinstance(carried_count, int) and carried_count >= 0
        assert isinstance(capture_count, int) and capture_count >= 0
        assert isinstance(scan_count, int) and scan_count >= 0
        self._sub_graph_name = sub_graph_name
        self._carried_count = carried_count
        self._capture_count = capture_count
        self._scan_count = scan_count

    # inputs: [cond, carries..., captures..., max_loop(optional)]
    # outputs: [carries..., scans...]
    # sub graph inputs: [iterator, cond, carries..., captures...]
    # sub graph outputs: [cond, carries..., scans...]
    def inference(self):
        assert len(self.input_names) >= 1 + self.carried_count + self.capture_count
        assert len(self.input_names) <= 2 + self.carried_count + self.capture_count
        assert len(self.output_names) == self.carried_count + self.scan_count

        assert self.inputs[0].is_value()
        assert self.inputs[0].dtype == ir.framework.DataType.BOOL
        if len(self.input_names) == 2 + self.carried_count + self.capture_count:
            assert self.inputs[-1].is_value()
            assert self.inputs[-1].is_integer()

        sub_graph = self.model.graphs[self.sub_graph_name]
        assert len(sub_graph.inputs) == 2 + self.carried_count + self.capture_count
        assert len(sub_graph.outputs) == 1 + self.carried_count + self.scan_count
        assert (
            sub_graph.get_symbol(sub_graph.inputs[0]).is_value()
            and sub_graph.get_symbol(sub_graph.inputs[0]).is_integer()
        )
        assert (
            sub_graph.get_symbol(sub_graph.inputs[1]).is_value()
            and sub_graph.get_symbol(sub_graph.inputs[1]).dtype
            == ir.framework.DataType.BOOL
        )

        for i in range(self.carried_count):
            sub_graph_symbol = sub_graph.get_symbol(sub_graph.outputs[i + 1])
            self.outputs[i].dtype = sub_graph_symbol.dtype
            self.outputs[i].shape = sub_graph_symbol.shape
            if sub_graph_symbol.is_tensor():
                self.outputs[i].stype = ir.framework.SymbolType.VARIABLE
            elif sub_graph_symbol.is_value():
                self.outputs[i].stype = ir.framework.SymbolType.VALUE
            elif sub_graph_symbol.is_vector():
                self.outputs[i].stype = ir.framework.SymbolType.VECTOR
            else:
                raise RuntimeError("Unknow error.")

        shape_name = self.graph.context.create_shape_name("loop")
        for i in range(self.scan_count):
            sub_graph_symbol = sub_graph.get_symbol(
                sub_graph.outputs[self.carried_count + i + 1]
            )
            self.outputs[self.carried_count + i].dtype = sub_graph_symbol.dtype
            if sub_graph_symbol.is_tensor():
                self.outputs[self.carried_count + i].shape = [
                    shape_name
                ] + sub_graph_symbol.shape
                self.outputs[
                    self.carried_count + i
                ].stype = ir.framework.SymbolType.VARIABLE
            elif sub_graph_symbol.is_value():
                self.outputs[self.carried_count + i].shape = [shape_name]
                self.outputs[
                    self.carried_count + i
                ].stype = ir.framework.SymbolType.VECTOR
            elif sub_graph_symbol.is_vector():
                self.outputs[self.carried_count + i].shape = [
                    shape_name
                ] + sub_graph_symbol.shape
                self.outputs[
                    self.carried_count + i
                ].stype = ir.framework.SymbolType.VARIABLE
            else:
                raise RuntimeError("Unknow error.")

    @property
    def sub_graph_name(self):
        return self._sub_graph_name

    @property
    def carried_count(self):
        return self._carried_count

    @property
    def capture_count(self):
        return self._capture_count

    @property
    def scan_count(self):
        return self._scan_count

    @property
    def attributes(self):
        return {
            "sub_graph_name": self.sub_graph_name,
            "carried_count": self.carried_count,
            "capture_count": self.capture_count,
            "scan_count": self.scan_count,
        }
