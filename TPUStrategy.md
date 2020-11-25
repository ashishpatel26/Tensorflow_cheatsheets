# TPU Strategy

---

### (Colab) Detect TPU

``` python3
import os
import tensorflow as tf

try:
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    
    print("Running on TPU", tpu.cluster_spec().as_dict()['worker'])
    print("Number of accelerators: {}".format(strategy.num_replicas_in_sync))

except ValueError:
    print("TPU failed to initialize.")
```
