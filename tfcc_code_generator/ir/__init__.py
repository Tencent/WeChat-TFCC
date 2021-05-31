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


def optimize(model, exclude_optimizers):
    from .optimizer import get_all_optimizers

    optimizers = get_all_optimizers()

    optimizer_names = set([optimizer.__class__.__name__ for optimizer in optimizers])

    if not exclude_optimizers:
        exclude_optimizers = set([])
    else:
        exclude_optimizers = set(exclude_optimizers)
        for name in exclude_optimizers:
            if name not in optimizer_names:
                raise RuntimeError(
                    "Invalid Optimizer name: {}. Valid name list: {}".format(
                        name, optimizer_names
                    )
                )

    optimizers = [
        optimizer
        for optimizer in optimizers
        if optimizer.__class__.__name__ not in exclude_optimizers
    ]

    changed = True
    while changed:
        changed = False
        for optimizer in optimizers:
            changed = changed or optimizer(model)
    return model


def fusion(model):
    from .fusionop import FusionOp

    f = FusionOp()
    for graph in model.graphs.values():
        f.process(graph)
    return model
