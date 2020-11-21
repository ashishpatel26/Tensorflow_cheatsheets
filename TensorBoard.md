# TensorBoard

https://www.tensorflow.org/tensorboard

``` python3
TensorBoard(log_dir = './logs', update_freq = 'epoch', **kwargs)
```

## Use TensorBoard callbacks

``` python3
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'log_dir')
model.fit(train_data, epochs = 5, callbacks = [tensorboard])
```

## Use TensorBoard in Google Colab

Load the extension
``` python3
%load_ext tensorboard
```

Run TensorBoard
``` python3
%tensorboard --logdir logs
```
