import tensorflow as tf
    
import numpy as np
import time

def loss(model, x, y):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def acc(model, dataset):
    epoch_loss = tf.keras.metrics.Mean()
    epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    loss_ = []
    acc_ = []
    for x, y in dataset:
        loss_value = loss(model, x, y)
        y_pred = model(x)

        epoch_loss(loss_value)
        epoch_acc(y, y_pred)

        loss_.append(epoch_loss.result())
        acc_.append(epoch_acc.result())

    loss_ = np.asarray(loss_).mean()
    acc_ = np.asarray(acc_).mean()

    return loss_, acc_


def train(model, train_dataset, test_dataset, epochs=1, vb=4, steps_per_epoch=390):
    optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)

    num_epochs = epochs
    print(f"Starting training.....")
    t = time.time()
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_dataset):
            print(f"{i}/{steps_per_epoch}")
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if(epoch%vb == 0):
            loss_, acc_ = acc(model, test_dataset)
            print(f"Time:{time.time()-t:.4f}s, Epoch:({epoch}/{num_epochs}), \
                    val_loss:{loss_:.4f}, val_acc:{acc_:.4f}")


    
def test():
    ''' Test the working of the cycling learning rate '''
    inputs = tf.keras.layers.Input((3,3,3,))
    d2 = tf.keras.layers.Conv2D(10,(3,3))(inputs)
    d3 = tf.keras.layers.Flatten()(d2)
    model = tf.keras.Model(inputs=inputs,outputs=d3)
    X_train, y_train = np.random.randn(1000,3,3,3), np.random.randint(0,10,size=(1000,))
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).batch(32)
    train(model, train_dataset, test_dataset, epochs=10, vb=3)

if __name__ == '__main__':
    test()
