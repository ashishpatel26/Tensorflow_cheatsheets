# GradientTape

``` python3

with tf.GraduentTape() as tape:
  logits = model(images, training = True)
  loss = loss_criteria(labels, logits)
  
loss_history.append(loss.numpy().mean())
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

```
