from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 #for our model
from tensorflow.keras.applications import ResNet152V2

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers

def build_MNV2(input_shape: tuple,
               alpha: int = 1,
               weights: list = None,
               dropout_per: float = 0.2,
               target_size: int = 168,
              learning_rate: float = 0.0002):
    mobilenetV2 = MobileNetV2(
                input_shape = input_shape,
                alpha = alpha,
                weights=weights,
                include_top=False)

    #Making all layers in MobilenetV2 trainable!
    for layer in mobilenetV2.layers:
        layer.trainable = True

    model = Sequential()
    model.add(mobilenetV2)
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation = 'relu'))
    model.add(layers.Dropout(dropout_per))
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(dropout_per))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dropout(dropout_per))
    model.add(layers.Dense(target_size, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )

    return model


def build_MNV2_reg(input_shape: tuple,
               alpha: int = 1,
               weights: list = None,
               dropout_per: float = 0.2,
               target_size: int = 168,
              learning_rate: float = 0.0002):
    mobilenetV2 = MobileNetV2(
                input_shape = input_shape,
                alpha = alpha,
                weights=weights,
                include_top=False)

    #Making all layers in MobilenetV2 trainable!
    for layer in mobilenetV2.layers:
        layer.trainable = True

    model = Sequential()
    model.add(mobilenetV2)
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Flatten())
    
    model.add(layers.Dense(1024, use_bias = False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_per))
    
    model.add(layers.Dense(512, use_bias = False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_per))
    
    model.add(layers.Dense(256, use_bias = False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_per))
    
    model.add(layers.Dense(target_size, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )

    return model

def build_ResNet_reg(input_shape: tuple,
               alpha: int = 1,
               weights: list = None,
               dropout_per: float = 0.2,
               target_size: int = 168,
              learning_rate: float = 0.0002):
    RESNET = ResNet152V2(
                input_shape = input_shape,
                weights=weights,
                include_top=False)

    #Making all layers in MobilenetV2 trainable!
    #for layer in RESNET.layers:
    #    layer.trainable = True

    model = Sequential()
    model.add(RESNET)
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Flatten())
    
    model.add(layers.Dense(1024, use_bias = False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_per))
    
    model.add(layers.Dense(512, use_bias = False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_per))
    
    model.add(layers.Dense(256, use_bias = False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_per))
    
    model.add(layers.Dense(target_size, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )

    return model  

def build_ResNet(input_shape: tuple,
               alpha: int = 1,
               weights: list = None,
               dropout_per: float = 0.2,
               target_size: int = 168,
              learning_rate: float = 0.0002):
    RESNET = ResNet152V2(
                input_shape = input_shape,
                weights=weights,
                include_top=False)

    #Making all layers in MobilenetV2 trainable!
    #for layer in RESNET.layers:
    #    layer.trainable = True

    model = Sequential()
    model.add(RESNET)
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation = 'relu'))

    model.add(layers.Dropout(dropout_per))
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(dropout_per))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dropout(dropout_per))
    model.add(layers.Dense(target_size, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )

    return model  

def build_minst_big(input_shape,
                dropout_per,
               target_size,
               learning_rate = 0.0002):
    
    mnst_model = Sequential([
    # This is the first convolution
    layers.Conv2D(128, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(2, 2),
    # The second convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # The third convolution
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # The forth Conco
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    layers.Flatten(),
    # 512 neuron hidden layer
    layers.Dense(512, activation='relu'),
    layers.Dropout(dropout_per),
        
#    layers.Dense(256, activation='relu'),
#    layers.Dropout(dropout_per),
        
    layers.Dense(target_size, activation='softmax')
    ])

    mnst_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )
    return mnst_model
    
def build_minst(input_shape,
                dropout_per,
               target_size,
               learning_rate = 0.0002):
    
    mnst_model = Sequential([
    # This is the first convolution
    layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    # The second convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # The third convolution
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.Conv2D(8, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    layers.Flatten(),
    # 512 neuron hidden layer
    layers.Dense(512, activation='relu'),
    layers.Dropout(dropout_per),
        
    layers.Dense(256, activation='relu'),
    layers.Dropout(dropout_per),
        
    layers.Dense(target_size, activation='softmax')
    ])

    mnst_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )
    return mnst_model


def build_minst2(input_shape,
                dropout_per,
               target_size,
               learning_rate = 0.0002):
    
    mnst_model = Sequential([
    # This is the first convolution
    layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    # The second convolution
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # The third convolution
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    layers.Flatten(),
    # 512 neuron hidden layer
    layers.Dense(512, activation='relu'),
    layers.Dropout(dropout_per),
        
    layers.Dense(256, activation='relu'),
    layers.Dropout(dropout_per),
        
    layers.Dense(target_size, activation='softmax')
    ])

    mnst_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )
    return mnst_model

def build_springer(input_shape,
                dropout_per,
               target_size,
               learning_rate = 0.0002,
                return_features = True):
    """Model based on this paper for character recognition
    https://link.springer.com/article/10.1007/s11036-019-01243-5
    """
    mnst_model = Sequential([
    # This is the first convolution
    layers.Conv2D(128, (5,5), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(2, 2),
    # The second convolution
    layers.Conv2D(128, (5,5), activation='relu'),
    layers.MaxPooling2D(2,2),
    # The third convolution
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # The Forth Layer
    layers.Conv2D(128, (3,3), activation='relu'),
    # Flatten the results to feed into a DNN
    layers.Flatten(),
    # 512 neuron hidden layer
    layers.Dense(512, activation='relu'),
    # drop out layer
    layers.Dropout(dropout_per),
    layers.Dense(target_size, activation='softmax')
    ])

    mnst_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )
    
    return mnst_model
def build_mini_minst(input_shape,
                dropout_per,
               target_size,
               learning_rate = 0.0002):
    
    mnst_model = Sequential([
    # This is the first convolution
    layers.Conv2D(20, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(2, 2),
    # The second convolution
    layers.Conv2D(10, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    # Third layer
    layers.Conv2D(5, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    layers.Flatten(),
    # 512 neuron hidden layer
    layers.Dense(256, activation='relu'),
    # drop out layer
    layers.Dropout(dropout_per),
    layers.Dense(target_size, activation='softmax')
    ])

    mnst_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=learning_rate),
        #optimizer = optimizers.RMSprop(learning_rate = learning_rate )    ,
        metrics=['accuracy']
        )
    return mnst_model