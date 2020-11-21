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
