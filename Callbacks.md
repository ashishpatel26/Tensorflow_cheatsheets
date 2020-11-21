# Callback classes


### Save the model checkpoint

``` python 3
model.fit(train_data, epochs = 5, validation_data = validation_data, verbose = 2,
          callbacks = [ModelCheckpoint('model.h5', verbose = 1)])
```

### Save the model weights

``` python 3
model.fit(train_data, epochs = 5, validation_data = validation_data, verbose = 2,
          callbacks = [ModelCheckpoint('model.h5', save_weights_only = True, verbose = 1)])
```

The filename can be customized, such as `'weights.{epoch:02d}-{val_loss:.2f}.h5'` to reflect the current metrics

### Save only the best checkpoint

``` python 3
model.fit(train_data, epochs = 5, validation_data = validation_data, verbose = 2,
          callbacks = [ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)])
```

### Early stopping

To prevent overfitting
``` python3
model.fit(train_data, epochs = 500, validation_data = validation_data,
          callbacks = [EarlyStopping(patience = 3, monitor = 'val_loss')])
```

The parameter `patience` specifies how many epochs to follow the best fit. To save the weights of that best fit, we can add `restore_best_weights = True`

``` python3
model.fit(train_data, epochs = 500, validation_data = validation_data,
          callbacks = [EarlyStopping(patience = 3, monitor = 'val_loss', restore_best_weights = True)])
```

### CSV logger

``` python3
model.fit(train_data, epochs = 500, validation_data = validation_data,
          callbacks = [CSVLogger('trainingLog.csv')])
```



### Custom callbacks

Example: display batch number and time
``` python3
import datetime

class MyCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs = None):
        print("Training begins: batch = {}, time = {}".format(batch, datetime.datetime.now().time()))
    
    def on_train_batch_end(self, batch, logs = None):
        print("Training ends: batch = {}, time = {}".format(batch, datetime.datetime.now().time()))

my_callback = MyCallback()
model.fit(train_data, epochs = 500, validation_data = validation_data,
          callbacks = [my_callback])
```

Example: early stopping after val_loss/train_loss goes beyond a specified threshold
``` python3
class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyCallback, self).__init__()
        self.threshold = threshold
    
    def on_epoch_end(self, epoch, logs = None):
        ratio = logs["val_loss"] / logs["loss"]
        print("Epoch {}: ratio = {:.2f}".format(epoch, ratio))
        
        if ratio > self.threshold:
            print("Ratio reached")
            self.model.stop_training = True

my_callback = MyCallback()
model.fit(train_data, epochs = 500, validation_data = validation_data,
          callbacks = [my_callback(threshold = 1.3)])
```

Callbacks can also be used to generate figures and save to disk (for examplem, when training GANs).

To save the image at the end of each epoch, and save an animated gif at the end of training
``` python3
import imageio

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, ...):
        ...
        self.images = []
        ...
    ...
    def on_epoch_end(self, epoch, logs = None):
        ...
        buffer = io.BytesIO()
        plt.savefig(buffer, format = 'png')
        buffer.seek(0)
        image = Image.open(buffer)
        self.images.append(np.array(image))
        
        if epoch % self.print_every == 0:
            plt.show()
    
    def on_train_end(self, logs = None):
        imageio.mimsave('animation.gif', self.imiages, fps = 1)
```
