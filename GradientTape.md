# GradientTape

``` python3

with tf.GraduentTape() as tape:
  logits = model(images, training = True)
  loss = loss_criteria(labels, logits)
  
loss_history.append(loss.numpy().mean())
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

```

Use `tape.watch` to watch a variable

``` python3
x = tf.ones((2,2))

with tf.GradientTape() as tape:
  tape.watch(x)
  y = tf.reduce_sum(x)
  z = tf.square(y)

grad = tape.gradient(z, x)

```

The tape will be discarded once the `tape.gradient()` method was called. To call `tape.gradient()` multiple times, we can set `persistent = True`

``` python3
with tf.GradientTape(persistent = True) as tape:
  ...
grad1 = tape.gradient(..., ...)
grad2 = tape.gradient(..., ...)
del tape
```
