```mermaid
graph TD
subgraph DST
    t_cast_y[base.Cast]
end
subgraph SRC
    cast_x[base.Cast] --> cast_y[base.Cast]
end
```