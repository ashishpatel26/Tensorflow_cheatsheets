# Mirrored Strategy for Multiple GPUs

---

### Initiate the strategy
``` python3
strategy = tf.distribute.MirroredStrategy()

print("Number of devices: {}".format(strategy.num_replicas_in_sync))
```

### Define batch size per replica

``` python
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
```

### Use Mirror Strategy

``` python3
with strategy.scope():
    model = tf.keras.Sequential([...])
    model.compile(loss = tf.keras.losses.SparseCategoticalCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE),
              optimizer = ..., metrics = ...)
    model.fit(train_data, epoch = 20)
```

### Create distributed datasets

``` python3
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


@tf.function
def dist_train_step(data):
    per_replica_loss = strategy.run(train_step, args = (data, ))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, asis = None)


def train_step(inputs):
    images, labels = inputs
    
    with tf.GradientTape() as tape:
        y_preds = model(images, training = True)
        loss = compute_loss(labels, y_preds)
        
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    
    train_acc_metric.update_state(labels, y_preds)
    return loss
```

### Other settings

1. The default setting of number of cores of each GPU is 8. If you have a GPU with less than 8 (such as 4) cores:

``` python3
import os

os.environ("TF_MIN_GPU_MULTIPROCESSOR_COUNT") = "4"
```

2. If your GPUs are with different models, set the `cross_device_ops` parameter when calling `tf.distribute.MirrirStrategy`

``` python3
strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())

print("Number of devices: {}".format(strategy.num_replicas_in_sync))
```
