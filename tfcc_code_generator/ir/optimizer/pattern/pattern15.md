Layer normalization pattern
```mermaid
graph TD
subgraph SRC0
SRC0_mean[math.ReduceMean] --> |0:1| SRC0_sub_0[math.Sub]
SRC0_sub_0 --> |0:0| SRC0_squared[math.Mul]
SRC0_sub_0 --> |0:1| SRC0_squared[math.Mul]
SRC0_squared --> SRC0_variance[math.ReduceMean]
SRC0_variance --> |0:0,1| SRC0_add_0[math.Add]
SRC0_add_0 --> SRC0_inv[math.Rsqrt]
SRC0_inv --> |0:0,1| SRC0_x[math.Mul]
SRC0_x --> |0:0,1| SRC0_ix[math.Mul]
SRC0_mean --> |0:0,1| SRC0_mx[math.Mul]
SRC0_x --> |0:0,1| SRC0_mx[math.Mul]
SRC0_ix --> |0:0,1| SRC0_add_1[math.Add]
SRC0_add_1 --> |0:0| SRC0_result[math.Sub]
SRC0_mx --> |0:1| SRC0_result
end

subgraph SRC1
SRC1_mean[math.ReduceMean] --> |0:1| SRC1_sub_0[math.Sub]
SRC1_sub_0 --> |0:0| SRC1_squared[math.Mul]
SRC1_sub_0 --> |0:1| SRC1_squared[math.Mul]
SRC1_squared --> SRC1_variance[math.ReduceMean]
SRC1_variance --> |0:0,1| SRC1_add_0[math.Add]
SRC1_add_0 --> SRC1_inv[math.Rsqrt]
SRC1_inv --> |0:0,1| SRC1_x[math.Mul]
SRC1_x --> |0:0,1| SRC1_ix[math.Mul]
SRC1_mean --> |0:0,1| SRC1_mx[math.Mul]
SRC1_x --> |0:0,1| SRC1_mx[math.Mul]
SRC1_ix --> |0:0| SRC1_sub_1[math.Sub]
SRC1_mx --> |0:1| SRC1_sub_1
SRC1_sub_1 --> |0:0,1| SRC1_result[math.Add]
end

subgraph SRC2
SRC2_mean[math.ReduceMean] --> |0:1| SRC2_sub_0[math.Sub]
SRC2_sub_0 --> |0:0| SRC2_squared[math.Mul]
SRC2_sub_0 --> |0:1| SRC2_squared[math.Mul]
SRC2_squared --> SRC2_variance[math.ReduceMean]
SRC2_variance --> |0:0,1| SRC2_add_0[math.Add]
SRC2_add_0 --> SRC2_inv[math.Rsqrt]
SRC2_inv --> |0:0,1| SRC2_x[math.Mul]
SRC2_x --> |0:0,1| SRC2_ix[math.Mul]
SRC2_mean --> |0:0,1| SRC2_mx[math.Mul]
SRC2_x --> |0:0,1| SRC2_mx[math.Mul]
SRC2_mx --> |0:1| SRC2_sub_1[math.Sub]
SRC2_sub_1 --> |0:0,1| SRC2_result[math.Add]
SRC2_ix --> |0:0,1| SRC2_result
end

subgraph SRC3
SRC3_mean[math.ReduceMean] --> |0:1| SRC3_sub_0[math.Sub]
SRC3_sub_0 --> |0:0| SRC3_squared[math.Mul]
SRC3_sub_0 --> |0:1| SRC3_squared[math.Mul]
SRC3_squared --> SRC3_variance[math.ReduceMean]
SRC3_variance --> |0:0,1| SRC3_add_0[math.Add]
SRC3_add_0 --> SRC3_inv[math.Sqrt]
SRC3_sub_0 --> |0:0| SRC3_mx[math.Div]
SRC3_inv --> |0:1| SRC3_mx[math.Div]
SRC3_mx --> |0:0,1| SRC3_x[math.Mul]
SRC3_x --> |0:0,1| SRC3_result[math.Add]
end
```