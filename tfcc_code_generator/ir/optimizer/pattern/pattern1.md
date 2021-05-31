```mermaid
graph TD
subgraph DST
    t_to_y[base.ToTensor/base.ToValue/base.ToVector]
end
subgraph SRC
    to_x[base.ToTensor/base.ToVector] --> to_y[base.ToTensor/base.ToValue/base.ToVector]
end
```