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

def _block(filters,strides=1,first=False):
  def f(inputs):
    x = inputs
    identity = x
    if(first == False):
      x = _br()(x)
    
    if(strides == 2):
      identity = tf.keras.layers.Conv2D(filters=filters,kernel_size=1,strides=2,
                                        use_bias=False,kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(5e-4)
                                        )(identity)
      #identity = _cbr(filters,kernel_size=1,strides=2)(identity)
      identity = tf.keras.layers.BatchNormalization(axis=-1)(identity)
    
    # Create a branch as follows: conv1->bn->relu->conv2 and add x + conv2
    x = _cbr(filters,strides=strides)(x)
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=3,
                               strides=1,padding='same',use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                               )(x)
    
    # Now addition operation
    x = tf.keras.layers.add([identity,x])

    return x
  return f

def _res_block(filters,strides=1):

  def f(inputs):
    identity = inputs
    x = inputs
    if strides==2:
      #identity = tf.keras.layers.Conv2D(filters,kernel_size=1,strides=2,
      #                                  using_bias=False,kernel_initializer=False)(identity)
      identity = _cbr(filters,kernel_size=1,strides=2)(identity)

    x = _cbr(filters,strides=strides)(x)
    x = _cbr(filters,strides=1)(x)

    x = tf.keras.layers.add([identity,x])

    return x
  return f

def _layer(filters,first=False,scaledown=True):

  def f(inputs):
    # Create two blocks 
    strides = 2
    if(scaledown == False):
      strides=1
    block_1 = _block(filters=filters,strides=strides)(inputs)
    block_2 = _block(filters=filters,strides=1)(block_1)

    return block_2
  
  return f

def Resnet18():
  inputs = tf.keras.layers.Input(shape=(32,32,3))
  conv_7x7 = _cbr(filters=64,kernel_size=7,strides=2)(inputs)
  layer_1 = _layer(filters=64,first=True,scaledown=False)(conv_7x7)
  layer_2 = _layer(filters=128)(layer_1)
  layer_3 = _layer(filters=256)(layer_2)
  layer_4 = _layer(filters=512)(layer_3)
  avg = tf.keras.layers.GlobalAveragePooling2D()(layer_4)
  outputs = tf.keras.layers.Dense(10,activation='softmax')(avg)
  
  model = tf.keras.Model(inputs=inputs,outputs=outputs)
  return model


if __name__ == "__main__":
    inputs = layers.Input(shape=(32,32,3))
    layer1 = _cbr(64,kernel_reg=tf.keras.regularizer.l2(5e-4))(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=layer1)
    print(model.summary())
