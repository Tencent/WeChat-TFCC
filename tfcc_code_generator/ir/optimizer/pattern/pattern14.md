```mermaid
graph TD
subgraph DST
    rsqrt[Rsqrt]
end
subgraph SRC
    sqrt[math.Sqrt] --> reciprocal[math.Reciprocal]
end
```