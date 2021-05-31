```mermaid
graph TD
subgraph DST
    t_create_vector[base.CreateVector]
end
subgraph SRC
    to_tensor[base.ToTensor] --> cast[base.Cast]
    cast --> to_vector[base.ToVector]
end
```