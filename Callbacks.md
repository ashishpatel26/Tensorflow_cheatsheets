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

### Save only the best checkpoint

``` python 3
model.fit(train_data, epochs = 5, validation_data = validation_data, verbose = 2,
          callbacks = [ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)])
```
