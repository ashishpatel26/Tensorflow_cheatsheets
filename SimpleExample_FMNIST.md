# Simple Example (Fashion-MNIST)

---

### Step 1: Define Network

``` python3
def Model():
    inputs = tf.keras.Input(shape = (784, ))
    x = tf.keras.layers.Dense(64, activation = 'relu')(inputs)
    x = tf.keras.layers.Dense(64, activatipn = 'relu')(x)
    outputs = tf.keras.layers.Dense(10, activation = 'softmax')(x)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    return model
```

### Step 2: Prepare Data Pipeline

``` python3
train_data = tfds.load('fashion_mnist', split = 'train')
test_data = tfds.load('fashion_mnist', split = 'test')

def preprocess_data(data):
    image = data["image"]
    image = tf.reshape(image, [-1])
    image = tf.cast(image, 'float32')
    image /= 225.0
    return image, data["label"]
  
train_data = train_data.map(preprocess_data)
test_data = test_data.map(preprocess_data)

batch_size = 64

train = train_data.shuffle(buffer_size = 1024).batch(batch_size)
test = test_data.batch(batch_size)
```

### Step 3: Define Loss, Optimizer, and Accuracy Metric

``` python3
loss_object = tf.keras.losses.SparseCategoricalCrossentropy() # label is integers instead of one-hot vectors

optimizer = tf.keras.optimizers.Adam()

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
```

### Step 4: Define Training Loop

``` python3

def apply_gradient(optimizer, model, X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = loss_object(y, y_pred)
        
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients,  model.trainable_weights))
    
    return y_pred, loss
        
def train_data_for_one_epoch():
    loss_list = []
    for train_X, train_y in train_data:
        y_pred, loss = apply_gradient(optimizer, model, train_X, train_y)
        loss_list.append(loss)
        train_acc_metric.update_state(train_y, y_pred)
    return loss_list

def model_validation():
    loss_list = []
    for val_X, val_y in test_data:
        y_pred = model(val_X)
        loss = loss_object(val_y, y_pred)
        loss_list.append(loss)
        val_acc_metric.update_state(val_y, y_pred)
    return loss_list

model = Model()

num_epochs = 20

for epoch in range(1, num_epochs+1):
    train_loss = train_data_for_one_epoch()
    val_loss = model_validation()
    
    train_loss_mean = np.mean(train_loss)
    val_loss_mean = np.mean(val_loss)
    
    train_acc = train_acc_metric.result()
    train_acc_metric.reset_states()
    
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
```
