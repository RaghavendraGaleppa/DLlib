import tensorflow as tf
from tensorflow.keras import layers

def _bn_relu():
    ''' Apply batchnorm and relu '''

    def f(inputs):

        bn = layers.BatchNormalization(axis=-1)(inputs)
        relu = layers.Activation('relu')(bn)

        return relu
    
    return f

def _cbr(filters:int, kernel_size=3, strides=1, *args):
    ''' Builds a Conv2D->BatchNorm->Relu block '''
    args = args or {}
    args.setdefault('kernel_init',None)
    args.setdefault('kernel_reg', None)

    def f(inputs):
        conv = layers.Conv2D(filters,kernel_size=kernel_size,
                            strides=strides, padding='same',
                            kernel_initializer=args['kernel_init'],
                            kernel_regularizer=args['kernel_reg'])(inputs)

        bn_relu = _bn_relu()(conv)

        return bn_relu
    return f

def _block(filters:int, kernel_size=3, strides=1, first=False,*args):
    pass

if __name__ == "__main__":
    inputs = layers.Input(shape=(32,32,3))
    layer1 = _cbr(64,kernel_reg=tf.keras.regularizer.l2(5e-4))(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=layer1)
    print(model.summary())