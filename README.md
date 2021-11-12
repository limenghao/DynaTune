# DynaTune: Dynamic Tensor Program Optimization in Deep Neural Network Compilation
This repository is the implementation of DynaTune [paper](https://openreview.net/pdf?id=GTGb3M_KcUl).
This folder dynatune includes all the files DynaTune needs.

## Requirements

Install TVM first. You can find TVM installation instructions [here](https://tvm.apache.org/docs/install/from_source.html).
**Note**: This project is based on TVM version in Feb/2021. You could find a project copy from [here](https://github.com/limenghao/incubator-tvm).
>Prepare llvm:
```
wget https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
tar xvJf clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz <path-to-llvm>
```

>Clone the TVM project from github:
```
git clone --recursive https://github.com/limenghao/incubator-tvm tvm
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
mkdir build
cp cmake/config.cmake build
```
>Edit build/config.cmake:
```
set(USE_LLVM <path-to-llvm>/bin/llvm-config)
set(USE_CUDA ON) (you can ignore this if you want to test cpu only)
```
>Building:
```
cd build
cmake ..
make -j6
```
>Add TVM into PYTHONPATH, edit your ~/.bashrc:
```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}
```
>Install other required packages:
```
pip install -r requirements.txt
```
>Add DynaTune files.
```
cp dynatune <path-to-tvm>/python/tvm/
cp tuner/tuner.py <path-to-tvm>/python/tvm/autotvm/tuner/
cp measure/measure_methods.py <path-to-tvm>/python/tvm/autotvm/measure/
```
>Install the packages used in pylearnpredictor.
```
pip install emcee  lmfit
```

## Classes introduction
- TaskState:
	Basic enitity class for DynaTune, save all middle-states of each task in the tuning.
- TaskScheduler: 
	Base class of tasks scheduler which allocate the time slices.
- RandomScheduler, RoundRobinScheduler: 
	Simple dynamic scheduler with random/roundrobin selecting strategy.
- TaskPredictor: 
	The model to fit the learning curve, which helps to calculate the potential gain of each tasks. It uses the models in the project [pylrpredictor](https://github.com/automl/pylearningcurvepredictor) with some changes to be usable for DynaTune.
- TaskSelector: 
	The strategy used to select the task among the tasks with their calculated potential gains.
- UCB1Selector
- MultiArmBanditScheduler: 
	The flexible scheduler with predictor and selector.

## Example

- import  packages.
```
import os
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime
from tvm.dynatune.scheduler import RandomTaskScheduler, RoundRobinScheduler,MultiArmBanditScheduler
```
- Get the symbol definition and random weight of a network.
```python
def get_network(name, batch_size):
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape
```
- Set up basic configuration
```python
target = "llvm" 
batch_size = 1
dtype = "float32"
model_name = "resnet-18"
log_file = "%s-cpu-random5hr.log" % model_name
input_name = "data"
tuning_option = {
    'log_filename': log_file,
    'tuner': 'xgb',
    'early_stopping': 50,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=500, repeat=1, max_converge_coef=0.1, timeout=100),
    ),
}
```
- Main function.
```python
def tune_and_evaluate(tuning_opt):
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))
    tscheduler = MultiArmBanditScheduler(tasks, 360, 20, **tuning_opt, predictor="ml")
    tscheduler.schedule()
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        module.set_input(**params)
        module.run()
        out = module.get_output(0)
        print(out)
        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=500, repeat=1)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
```
- Call the main function.
```python
tune_and_evaluate(tuning_option)
```
### End