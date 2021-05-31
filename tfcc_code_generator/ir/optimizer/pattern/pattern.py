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

import os
import re
import time
import enum
import logging
import inspect
import ir.framework
import ir.node
from ir.optimizer.patternmanager import NodeManager
from ir.optimizer.patterngroup import PatternGroup


class PatternRule(object):
    def __init__(self, type_map: dict, links: dict, end_name: str):
        self._type_map = type_map
        self._links = links
        self._end_name = end_name

    @property
    def type_map(self):
        return self._type_map

    @property
    def links(self):
        return self._links

    @property
    def end_name(self):
        return self._end_name


class Pattern(object):
    _mermaid_re = re.compile(
        r"^\s*(\w+)([\[\(\{][\w\./]+[\]\)\}]){0,1}\s*-->\s*(\|\d+:[\d,]+\|){0,1}\s*(\w+)([\[\(\{][\w\./]+[\]\)\}]){0,1}\s*$"
    )
    _mermaid_re2 = re.compile(r"^\s*(\w+)([\[\(\{][\w\./]+[\]\)\}]){0,1}\s*$")

    def __init__(self):
        self._rule_map = {}

        markdown = open(self.markdown_path, "r").read()
        matchs = re.findall(r"```mermaid\n([\s\S]+?)\n```", markdown)
        if len(matchs) != 1:
            raise RuntimeError("Markdown format error")
        mermaid = matchs[0]

        matchs = re.findall("subgraph (SRC\w*?)\n([\s\S]+?)\nend", mermaid)
        graph_desc_map = {}
        for rule_name, desc in matchs:
            if rule_name not in graph_desc_map:
                graph_desc_map[rule_name] = ""
            graph_desc_map[rule_name] += "\n" + desc

        for rule_name in graph_desc_map:
            self.parse_lines(rule_name, graph_desc_map[rule_name].split("\n"), {}, {})

    def parse_lines(self, rule_name: str, lines: list, type_map: dict, links: dict):
        if not lines:
            idx = 0
            while True:
                if idx == 0:
                    real_rule_name = rule_name
                else:
                    real_rule_name = rule_name + ":" + str(idx)
                idx += 1
                if real_rule_name in self._rule_map:
                    continue
                end_name = None
                names = set([name for name, _ in links.values()])
                for name in type_map.keys():
                    if name not in names:
                        if end_name:
                            raise RuntimeError("Output count of the graph unequal to 1")
                        end_name = name

                if not end_name:
                    raise RuntimeError("Output count of the graph unequal to 1")
                self._rule_map[real_rule_name] = PatternRule(type_map, links, end_name)
                return

        line = lines[0]
        if re.match(r"^\s*$", line):
            return self.parse_lines(rule_name, lines[1:], type_map.copy(), links.copy())
        src, dst, types = self.parse_mermaid(line)
        type_map.update(types)
        if not src or not dst:
            return self.parse_lines(rule_name, lines[1:], type_map.copy(), links.copy())
        succ = False
        for slot in dst[1]:
            if (dst[0], slot) in links:
                continue
            succ = True
            new_links = links.copy()
            new_links[(dst[0], slot)] = src
            self.parse_lines(rule_name, lines[1:], type_map.copy(), new_links)
        if not succ:
            raise RuntimeError("Topology error")

    def parse_mermaid(self, line):
        matchs = re.findall(self._mermaid_re, line)
        type_map = {}
        if len(matchs) != 1:
            return self.parse_mermaid_v2(line)
        src, src_type, slots, dst, dst_type = matchs[0]
        if not src or not dst:
            raise RuntimeError("Mermaid format error")
        if src_type:
            src_type = src_type[1:-1]
            type_map[src] = self.get_node_cls(src_type)
        if dst_type:
            dst_type = dst_type[1:-1]
            type_map[dst] = self.get_node_cls(dst_type)
        if slots:
            src_slot, dst_slot = slots[1:-1].split(":")
            if "," in dst_slot:
                slots = [int(src_slot), [int(s) for s in dst_slot.split(",")]]
            else:
                slots = [int(src_slot), [int(dst_slot)]]
        else:
            slots = [0, [0]]
        return (src, slots[0]), (dst, slots[1]), type_map

    def parse_mermaid_v2(self, line):
        matchs = re.findall(self._mermaid_re2, line)
        if len(matchs) != 1:
            raise RuntimeError("Mermaid format error")
        src, src_type = matchs[0]
        if not src or not src_type:
            raise RuntimeError("Mermaid format error")
        src_type = src_type[1:-1]
        type_map = {src: self.get_node_cls(src_type)}
        return None, None, type_map

    def get_node_cls(self, uris):
        cls_list = []
        for uri in uris.split("/"):
            if not uri:
                continue
            value = ir.node
            for name in uri.split("."):
                if not hasattr(value, name):
                    raise RuntimeError("Node class ir.node.{} not found".format(name))
                value = getattr(value, name)
            if not isinstance(value, type):
                raise RuntimeError("Object ir.node.{} is not a type".format(name))
            if not issubclass(value, ir.node.Node):
                raise RuntimeError(
                    "Object ir.node.{} is not a sub class of ir.node.Node".format(name)
                )
            cls_list.append(value)
        if not cls_list:
            raise RuntimeError("Empty node cls list")
        return tuple(cls_list)

    @property
    def markdown_path(self):
        path = inspect.getfile(self.__class__)
        name = os.path.basename(path)
        assert name[-3:] == ".py"
        name = name[:-3] + ".md"
        path = os.path.dirname(path)
        return os.path.join(path, name)

    def process(self, node_manager: NodeManager):
        ts = time.time()
        process_count = 0
        may_useless_nodes = set()
        for rule_name in self._rule_map:
            pc, nodes = self.process_rule(node_manager, rule_name)
            process_count += pc
            may_useless_nodes.update(nodes)

        if process_count > 0:
            node_manager.graph.reflash_symbols()
            logging.debug(
                "{} process {} cost: {:.4} s".format(
                    self.__class__.__name__, process_count, time.time() - ts
                )
            )
            return may_useless_nodes
        else:
            return None

    def process_rule(self, node_manager: NodeManager, rule_name: str):
        rule = self._rule_map[rule_name]
        process_count = 0
        may_useless_nodes = set()
        may_end_nodes = set()
        for node_cls in rule.type_map[rule.end_name]:
            if node_cls in node_manager.type_map:
                may_end_nodes.update(node_manager.type_map[node_cls])
        for end_node in sorted(may_end_nodes, key=lambda a: a.name):
            node_map, matched_rule_set = self.check_pattern_node(
                node_manager, rule, {rule.end_name: end_node}, rule.end_name
            )
            if node_map is None:
                continue
            if len(matched_rule_set) != len(rule.links):
                continue
            if self.process_pattern(node_manager, rule_name, node_map.copy()):
                process_count += 1
                may_useless_nodes.update(node_map.values())
        return process_count, may_useless_nodes

    def check_pattern_node(
        self,
        node_manager: NodeManager,
        rule: PatternRule,
        node_map: dict,
        node_alias: str,
    ):
        matched_rule_set = set()
        for i, name in enumerate(node_map[node_alias].input_names):
            if not (node_alias, i) in rule.links:
                continue
            # may constant
            if not name in node_manager.name_map:
                return None, set()
            parent_node = node_manager.name_map[name]
            parent_alias, parent_slot = rule.links[(node_alias, i)]
            if not isinstance(parent_node, rule.type_map[parent_alias]):
                return None, set()
            if parent_node.output_names[parent_slot] != name:
                return None, set()
            if parent_alias in node_map and node_map[parent_alias] != parent_node:
                return None, set()
            node_map[parent_alias] = parent_node
            node_map, new_matched_rule_set = self.check_pattern_node(
                node_manager, rule, node_map, parent_alias
            )
            if node_map is None:
                return None, set()
            matched_rule_set.add(((node_alias, i), rule.links[(node_alias, i)]))
            matched_rule_set.update(new_matched_rule_set)
        return node_map, matched_rule_set

    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        raise NotImplementedError

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_0
