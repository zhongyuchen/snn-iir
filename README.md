# snn-iir

PyTorch implementation of IJCAI 2020 paper Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network
[[arXiv]](https://arxiv.org/abs/2003.02944) [[IJCAI 2020]](https://www.ijcai.org/Proceedings/2020/388)

## Prerequisites

Install all the required Python packages:
```
pip install -r requirements.txt
```

## Usage

### Train Model

Run Python script to train the corresponding model:
```
python *.py --train
```

### Test Model

#### Prepare Trained Weights

Do one of the following:
1. Use Trained Weights
    * Move to appropriate path.
2. Use ZIP Weights
    * Download the ZIP weights from __Results__ section;
    * Unzip ZIP files to get trained weights;
    * Move to appropriate path;
    * Modify `test_checkpoint_path` in `.yaml` config file;
    * Run Python script to test the corresponding model with assigned weights:

#### Test Model

* Modify `test_checkpoint_path` in `.yaml` config file;
* Run Python script to test the corresponding model with assigned weights:
```
python *.py --test
```

## Models

Details of the models for the following 3 tasks.

### Associative Memory

|experiment|network|states|filter|dataset|encoding|length|
|----------|-------|------|------|-------|--------|------|
|associative_memory|MLP|zero|dual exp iir|Pattern Dataset|original|300|


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

~~Not implemented.~~

## Results

The results of the following 3 tasks.

### Associative Memory

|experiment|train|dev|test|ZIP weights|paper|
|----------|-----|---|----|-------|-----|
|associative_memory|0.0031(93)|0.00369(92)|0.0042(92)|[92](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/associative_memory_checkpoint_92.zip)|-|

### Vision Tasks

|experiment|train|dev|test|ZIP weights|paper|
|----------|-----|---|----|-------|-----|
|snn_mlp_1|99.252(72)|98.58(72)|98.94(72)|[72](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_mlp_1_72_20210111-051825.zip)|-|
|snn_mlp_1_non_zero|99.116(93)|98.488(93)|98.858(93)|[93](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_mlp_1_non_zero_93_20210111-145608.zip)|-|
|snn_mlp_1_poisson_input|99.208(98)|98.628(98)|98.928(98)|[98](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_mlp_1_poisson_input_98_20210112-020238.zip)|-|
|snn_mlp_2|99.3(72)|98.66(72)|98.96(72)|[72](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_mlp_2_72_20210111-041817.zip)|-|
|snn_mlp_2_poisson_input|99.284(96)|98.748(96)|98.978(96)|[96](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_mlp_2_poisson_input_96_20210112-004922.zip)|-|
|snn_conv_1_mnist|99.84(99)|99.47(99)|__99.59__(99)|[99](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_conv_1_mnist_99_20210112-050334.zip)|-|
|snn_conv_1_mnist_poisson_input|99.822(93)|99.479(93)|__99.519__(93)|[93](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_conv_1_mnist_poisson_input_93_20210113-044956.zip)|99.46|
|snn_conv_1_nmnist|99.998(51)|98.708(89)|98.558(89)|[89](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_conv_1_nmnist_89_20210115-131718.zip)|__99.39__|
|snn_conv_1_gesture|95.474(46)|85.156(46)|66.319(46)|[46](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_conv_1_gesture_46_20210124-140846.zip)|__96.09__|
|snn_conv_1_gesture_30|96.094(59)|85.938(59)|68.75(59)|[59](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_conv_1_gesture_30_59_20210124-131913.zip)|__96.09__|
|snn_conv_1_gesture_max|97.845(68)|75.781(68)|70.486(68)|[68](https://github.com/zhongyuchen/snn-iir/releases/download/v1.0.0/checkpoint_snn_conv_1_gesture_max_68_20210125-122242.zip)|__96.09__|

### Times Series Classification

~~Not implemented.~~

## Author

Zhongyu Chen
