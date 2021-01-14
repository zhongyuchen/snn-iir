# snn-iir

```
python snn_mlp_1.py --train
python snn_mlp_1_non_zero.py --train
python snn_mlp_1_poisson_input.py --train
python snn_mlp_2.py --train
python snn_mlp_2_poisson_input.py --train
```

## Models

### Associative Memory

### Vision Tasks

|experiment|network|states|filter|dataset|encoding|
|----------|-------|------|------|-------|--------|
|snn_mlp_1|MLP|zero|dual exp iir|MNIST|copy along time dimension|
|snn_mlp_1_non_zero|MLP|preserved|dual exp iir|MNIST|copy along time dimension|
|snn_mlp_1_poisson_input|MLP|zero|dual exp iir|MNIST|rate-based poisson|
|snn_mlp_2|MLP|zero|first order low pass|MNIST|copy along time dimension|
|snn_mlp_2_poisson_input|MLP|zero|first order low pass|MNIST|rate-based poisson|
|snn_conv_1_mnist|CNN|zero|dual exp iir|MNIST|copy along time dimension|
|snn_conv_1_mnist_poisson_input|CNN|zero|dual exp iir|MNIST|rate-based poisson|
|snn_conv_1_nmnist|CNN|zero|dual exp iir|N-MNIST|accumulate within time window(OR)|
|snn_conv_1_gesture|CNN|zero|dual exp iir|DVS128 Gesture Dataset|accumulate within time window(OR)|

### Times Series Classification

## Results

### Associative Memory

|experiment|train|dev|test|paper|
|----------|-----|---|----|-----|
|associative_memory|0.0031(93)|0.00369(92)|0.0042(92)|-|

### Vision Tasks

|experiment|train|dev|test|paper|
|----------|-----|---|----|-----|
|snn_mlp_1|99.252(72)|98.58(72)|98.94(72)|-|
|snn_mlp_1_non_zero|99.116(93)|98.488(93)|98.858(93)|-|
|snn_mlp_1_poisson_input|99.208(98)|98.628(98)|98.928(98)|-|
|snn_mlp_2|99.3(72)|98.66(72)|98.96(72)|-|
|snn_mlp_2_poisson_input|99.284(96)|98.748(96)|98.978(96)|-|
|snn_conv_1_mnist|99.84(99)|99.47(99)|99.59(99)|-|
|snn_conv_1_mnist_poisson_input|99.822(93)|99.479(93)|99.519(93)|99.46|
|snn_conv_1_nmnist|-|-|-|-|
|snn_conv_1_gesture|-|-|-|-|

### Times Series Classification

