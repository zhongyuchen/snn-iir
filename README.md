# snn-iir

PyTorch implementation of Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network (IJCAI 20)

## prerequisites

```
pip install -r requirements.txt
```

## run code

### train model

```
python *.py --train
```

### test model

```
python *.py --test
```

## Models

### Associative Memory

### Vision Tasks

|experiment|network|states|filter|dataset|encoding|length|
|----------|-------|------|------|-------|--------|------|
|snn_mlp_1|MLP|zero|dual exp iir|MNIST|copy along time dimension|25|
|snn_mlp_1_non_zero|MLP|preserved|dual exp iir|MNIST|copy along time dimension|25|
|snn_mlp_1_poisson_input|MLP|zero|dual exp iir|MNIST|rate-based poisson|25|
|snn_mlp_2|MLP|zero|first order low pass|MNIST|copy along time dimension|25|
|snn_mlp_2_poisson_input|MLP|zero|first order low pass|MNIST|rate-based poisson|25|
|snn_conv_1_mnist|CNN|zero|dual exp iir|MNIST|copy along time dimension|25|
|snn_conv_1_mnist_poisson_input|CNN|zero|dual exp iir|MNIST|rate-based poisson|25|
|snn_conv_1_nmnist|CNN|zero|dual exp iir|N-MNIST|accumulate within time window(OR)|30|
|snn_conv_1_gesture|CNN|zero|dual exp iir|DVS128 Gesture Dataset|accumulate within time window(OR)|50|
|snn_conv_1_gesture_30|CNN|zero|dual exp iir|DVS128 Gesture Dataset|accumulate within time window(OR)|30|
|snn_conv_1_gesture_max|CNN|zero|dual exp iir|DVS128 Gesture Dataset|accumulate within time window(SUM)/frame(MAX)|30|

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
|snn_conv_1_mnist|99.84(99)|99.47(99)|__99.59__(99)|-|
|snn_conv_1_mnist_poisson_input|99.822(93)|99.479(93)|__99.519__(93)|99.46|
|snn_conv_1_nmnist|99.998(51)|98.708(89)|98.558(89)|__99.39__|
|snn_conv_1_gesture|95.474(46)|85.156(46)|66.319(46)|__96.09__|
|snn_conv_1_gesture_30|96.094(59)|85.938(59)|68.75(59)|__96.09__|
|snn_conv_1_gesture_max|97.845(68)|75.781(68)|70.486(68)|__96.09__|

### Times Series Classification

