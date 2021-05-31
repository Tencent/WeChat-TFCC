## Convert math.Pow to math.Mul

```mermaid
graph TD
subgraph DST
    t_mul[math.Mul] --> t_other[math.Mul...]
end
subgraph SRC
    pow[math.Pow]
end
```
