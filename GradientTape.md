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
