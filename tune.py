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

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
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


# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
target = "llvm"
# target = "cuda"
batch_size = 1
dtype = "float32"
model_name = "resnet-18"
# model_name = "vgg-16"
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

########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))
    tasks = tasks[:3]
    # print("tasks:" + str(tasks))
    print("tasks: %d" % len(tasks))
    # run tuning tasks
    # tune_kernels(tasks, **tuning_opt)
    
    # tscheduler = dynatune.Scheduler("Random", tasks, 3600*5, 100, **tuning_opt)
    # tscheduler = RandomTaskScheduler(tasks, 360, 20, **tuning_opt)
    # tscheduler = RoundRobinScheduler(tasks, 360, 20, **tuning_opt)
    tscheduler = MultiArmBanditScheduler(tasks, 360, 20, **tuning_opt, predictor="ml")
    tscheduler.schedule()
    # compile kernels with graph-level best records
    # log_file = "logs/for-inference/vgg-16-cpu-ada-new.log"
    # with autotvm.apply_history_best(log_file):
    if True:
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # upload parameters to device
        ctx = tvm.cpu()
        # ctx = tvm.gpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        module.set_input(**params)
        module.run()
        out = module.get_output(0)
        # print(out)
        # evaluate
        # print("Evaluate inference time cost...")
        # ftimer = module.module.time_evaluator("run", ctx, number=500, repeat=1)
        # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        # print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        #       (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)
