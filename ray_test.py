import ray
import platform

ray.init(address='auto')

@ray.remote(num_gpus=1, resources={"arch_x86": 0.999, "gpu_type_cuda": 1})
def simple_math(num):
    import tensorflow as tf
    x = tf.constant(num)
    tf.print(x, x.device)
    return f"Hello from {platform.machine()}! 1+1={2}"

print("Sending simple task...")
x = float('32.0')
future = simple_math.remote(x)
print(ray.get(future))
