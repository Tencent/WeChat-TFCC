```mermaid
graph TD
subgraph DST
    t_cast[base.CreateVector] --> to_tensor[base.ToTensor]
end
subgraph SRC0
    SRC0_to_tensor[base.ToTensor] --> SRC0_cast[base.Cast]
    SRC0_cast --> SRC0_gather_or_slice[nn.Gather/base.SliceV2/base.SliceV1]
    SRC0_gather_or_slice --> |0:0| SRC0_concat[base.Concat]
    SRC0_concat --> SRC0_cast2[base.Cast]
end
subgraph SRC1
    SRC1_to_tensor[base.ToTensor] --> SRC1_gather_or_slice[nn.Gather/base.SliceV2/base.SliceV1]
    SRC1_gather_or_slice --> |0:0| SRC1_concat[base.Concat]
    SRC1_concat --> SRC1_cast2[base.Cast]
end
subgraph SRC2
    SRC2_to_tensor[base.ToTensor] --> SRC2_cast[base.Cast]
    SRC2_cast --> SRC2_gather_or_slice[nn.Gather/base.SliceV2/base.SliceV1]
    SRC2_gather_or_slice --> |0:1| SRC2_concat[base.Concat]
    SRC2_concat --> SRC2_cast2[base.Cast]
end
subgraph SRC3
    SRC3_to_tensor[base.ToTensor] --> SRC3_gather_or_slice[nn.Gather/base.SliceV2/base.SliceV1]
    SRC3_gather_or_slice --> |0:1| SRC3_concat[base.Concat]
    SRC3_concat --> SRC3_cast2[base.Cast]
end
subgraph SRC4
    SRC4_to_tensor[base.ToTensor] --> SRC4_cast[base.Cast]
    SRC4_cast --> SRC4_gather_or_slice[nn.Gather/base.SliceV2/base.SliceV1]
    SRC4_gather_or_slice --> |0:2| SRC4_concat[base.Concat]
    SRC4_concat --> SRC4_cast2[base.Cast]
end
subgraph SRC5
    SRC5_to_tensor[base.ToTensor] --> SRC5_gather_or_slice[nn.Gather/base.SliceV2/base.SliceV1]
    SRC5_gather_or_slice --> |0:2| SRC5_concat[base.Concat]
    SRC5_concat --> SRC5_cast2[base.Cast]
end
subgraph SRC6
    SRC6_to_tensor[base.ToTensor] --> SRC6_cast[base.Cast]
    SRC6_cast --> SRC6_gather_or_slice[nn.Gather/base.SliceV2/base.SliceV1]
    SRC6_gather_or_slice --> |0:3| SRC6_concat[base.Concat]
    SRC6_concat --> SRC6_cast2[base.Cast]
end
subgraph SRC7
    SRC7_to_tensor[base.ToTensor] --> SRC7_gather_or_slice[nn.Gather/base.SliceV2/base.SliceV1]
    SRC7_gather_or_slice --> |0:3| SRC7_concat[base.Concat]
    SRC7_concat --> SRC7_cast2[base.Cast]
end

```