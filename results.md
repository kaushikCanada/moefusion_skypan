# Results


## Table 2: Label Efficiency (mIoU at different training data fractions)

| Model | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| UNet-EffB4 |0.1995 |0.2978 |0.6934 |0.7320 |0.7531 |
| U-Panopticon | 0.7024|0.7396 |0.7632 |0.7867 | 0.7948|
| SegFormer-B2 | 0.4759| 0.5301| 0.6030| 0.6471| 0.6957|
| SegFormer+Pan | 0.7342| 0.7603| 0.7769| 0.7924| 0.8017|

## Table 3: Convergence (val mIoU at epoch)

| Model | 10 | 20 | 30 |
|---|---|---|---|
| UNet-EffB4 | 0.6858| 0.7265| 0.7531|
| U-Panopticon | 0.7772| 0.7823| 0.7948|
| SegFormer-B2 | 0.6232| 0.6756| 0.6957|
| SegFormer+Pan | 0.7950| 0.7966| 0.8017|


