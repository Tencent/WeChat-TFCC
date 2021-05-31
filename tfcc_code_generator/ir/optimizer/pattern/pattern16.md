```mermaid
graph TD
subgraph SRC
equal[relation.Equal] --> not[relation.Not]
end

subgraph DST
not_equal[relation.UnEqual]
end
```