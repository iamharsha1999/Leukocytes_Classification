from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout, Input, Activation, BatchNormalization, Flatten, GlobalMaxPooling2D, Add, SeparableConv2D, Flatten
from keras.models import Sequential, Model
from keras import applications
from keras import backend as K


class DL_Model:


    @staticmethod
    def build_feature_model( height, width, depth, nc, model = 'resnet50'):
        ## Decide the Input Shape
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1
        else:
            input_shape = (height, width, depth)
            channel_dim = -1

        ## Build the ResNet50
        if model == 'resnet50':

            base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= input_shape)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.7)(x)
            x = Dense(32, activation = 'relu')(x)
            x = BatchNormalization()(x)
            x = Dense(16, activation = 'relu')(x)
            x = BatchNormalization()(x)
            predictions = Dense(nc, activation= 'softmax')(x)
            model = Model(inputs = base_model.input, outputs = predictions)


            return model

        elif model == 'densenet':

            base_model = applications.densenet.DenseNet121(weights = None, include_top = False, input_shape = input_shape)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(32, activation = "relu")(x)
            x = BatchNormalization()(x)
            x = Dense(16, activation = "relu")(x)
            x = BatchNormalization()(x)
            predictions = Dense(nc, activation = 'softmax')(x)
            model = Model(inputs = base_model.input, outputs = predictions)

            return model

        ## Build the SmallVGG16
        elif model == 'smallvgg16':


            # first (and only) CONV => RELU => POOL block
            inpt = Input(shape = input_shape)
            x = Conv2D(32, (3, 3), padding = "same")(inpt)
            x = Activation("relu")(x)
            x = BatchNormalization(axis = channel_dim)(x)
            x = MaxPooling2D(pool_size = (3, 3))(x)

            # first CONV => RELU => CONV => RELU => POOL block
            x = Conv2D(64, (3, 3), padding = "same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis = channel_dim)(x)
            x = Conv2D(64, (3, 3), padding = "same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis = channel_dim)(x)
            x = MaxPooling2D(pool_size = (2, 2))(x)

            # second CONV => RELU => CONV => RELU => POOL Block
            x = MaxPooling2D(pool_size = (2, 2))(x)

            # first (and only) FC layer
            x = Flatten()(x)
            x = Dense(64, activation = 'relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            x = Dense(32, activation = 'relu')(x)
            x = BatchNormalization()(x)

            x = Dense(nc, activation = 'softmax')(x)
            model  = Model(inputs=inpt, outputs = x)

            return model

        elif model == 'VGG16':

            base_model = applications.vgg16.VGG16(weights= None, include_top=False, input_shape= input_shape)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.7)(x)
            predictions = Dense(nc, activation= 'softmax')(x)
            model = Model(inputs = base_model.input, outputs = predictions)

            return model

        elif model == 'xception':

            base_model = applications.xception.Xception(weights= None, include_top=False, input_shape= input_shape)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.7)(x)
            predictions = Dense(nc, activation= 'softmax')(x)
            model = Model(inputs = base_model.input, outputs = predictions)

            return model

        elif model == 'inceptionv3':

            base_model = applications.inception_v3.InceptionV3(weights= None, include_top=False, input_shape= input_shape)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.7)(x)
            predictions = Dense(nc, activation= 'softmax')(x)
            model = Model(inputs = base_model.input, outputs = predictions)

            return model

        elif model == 'hegde_cnn':

            inpt = Input(shape = input_shape)

            # Layer 1
            x = Conv2D(96, (11, 11), strides = 4, padding = "valid")(inpt)
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size = (3,3), strides = 2)

            # Layer 2
            x = Conv2D(256, (5,5), strides = 1, padding = "same")(x)
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size = (3,3), strides = 2)

            #Layer 3
            x = Conv2D(384, (3,3), strides = 1, padding = "same")(x)
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size = (3,3), padding = "same")(x)

            #Layer 4
            x = Conv2D(384, (3,3), strides = 1, padding = "same")(x)
            x = Activation("relu")(x)


            # Layer 5
            x = Conv2D(256, (3,3), strides = 1, padding = "same")(x)
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size = (3,3), padding = "same")(x)

            # Fully Connected Layer
            x = Dense(500, activation = "relu")(x)
            x = Dense(256, activation = "relu")(x)
            x = Dense(5, activation = "softmax")(x)

    
