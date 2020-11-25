# Mirrored Strategy

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
```

### Compile and train the model as usual

``` python3
model.compile(loss = ..., optimizer = ..., metrics = ...)
model.fit(train_data, epoch = 20)
```
