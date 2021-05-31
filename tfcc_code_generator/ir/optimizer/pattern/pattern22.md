Gelu pattern
```mermaid
graph TD
subgraph DST
    t_identity[math.Identity]
end
subgraph SRC0
    SRC0_transpose_1[base.Transpose] --> SRC0_transpose_2[base.Transpose]
end
```