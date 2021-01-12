# snn-iir

```
python snn_mlp_1.py --train
python snn_mlp_1_non_zero.py --train
python snn_mlp_1_poisson_input.py --train
python snn_mlp_2.py --train
python snn_mlp_2_poisson_input.py --train
```

|experiment|network|states|filter|dataset|encoding|train|dev|test|paper|
|----------|-------|------|------|-------|--------|-----|---|----|-----|
|snn_mlp_1|MLP|zero|dual exp iir|MNIST|copy along time dimension|-|-|-|-|
|snn_mlp_1_non_zero|MLP|preserved|dual exp iir|MNIST|copy along time dimension|-|-|-|-|
|snn_mlp_1_poisson_input|MLP|zero|dual exp iir|MNIST|rate-based poisson|-|-|-|-|
|snn_mlp_2|MLP|zero|first order low pass|MNIST|copy along time dimension|-|-|-|-|
|snn_mlp_2_poisson_input|MLP|zero|first order low pass|MNIST|rate-based poisson|-|-|-|-|
|snn_conv_1_mnist|CNN|zero|dual exp iir|MNIST|copy along time dimension|-|-|-|-|
|snn_conv_1_poisson_input|CNN|zero|dual exp iir|MNIST|rate-based poisson|-|-|-|-|


