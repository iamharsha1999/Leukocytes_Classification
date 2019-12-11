from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dropout, Input, Activation, BatchNormalization, Flatten, GlobalMaxPooling2D
from keras.models import Sequential, Model
from keras import applications
from keras import backend as K

class DL_Model:


    @staticmethod
    def build_feature_model( height, width, depth, nc, model = 'resnet50',):

        ## Build the ResNet50
        if model == 'resnet50':
            if K.image_data_format() == "channels_first":
                input_shape = (depth, height, width)
                channel_dim = 1
            else:
                input_shape = (height, width, depth)
                channel_dim = -1

            base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= input_shape)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.7)(x)
            predictions = Dense(nc, activation= 'softmax')(x)
            model = Model(inputs = base_model.input, outputs = predictions)


            return model

        ## Build the SmallVGG16
        elif model == 'smallvgg16':
            if K.image_data_format() == "channels_first":
                input_shape = (depth, height, width)
                channel_dim = 1
            else:
                input_shape = (height, width, depth)
                channel_dim = -1

            # first (and only) CONV => RELU => POOL block
            inpt = Input(shape = input_shape)
            x = Conv2D(32, (3, 3), padding = "same")(inpt)
            x = Activation("relu")(x)
            x = BatchNormalization(axis = channel_dim)(x)
            x = MaxPooling2D(pool_size = (3, 3))(x)
            # x = Dropout(0.25)(x)

            # first CONV => RELU => CONV => RELU => POOL block
            x = Conv2D(64, (3, 3), padding = "same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis = channel_dim)(x)
            x = Conv2D(64, (3, 3), padding = "same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis = channel_dim)(x)
            x = MaxPooling2D(pool_size = (2, 2))(x)
            # x = Dropout(0.25)(x)

            # second CONV => RELU => CONV => RELU => POOL Block
            x = Conv2D(128, (3, 3), padding = "same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis = channel_dim)(x)
            x = Conv2D(128, (3, 3), padding = "same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis = channel_dim)(x)
            x = MaxPooling2D(pool_size = (2, 2))(x)
            # x = Dropout(0.25)(x)

            # first (and only) FC layer
            x = GlobalMaxPooling2D()(x) # Change to GlobalMaxPooling2D
            x = Dense(64, activation = 'swish')(x)
            x = BatchNormalization(axis = channel_dim)(x)
            # x = Dropout(0.5)(x)
            x = Dense(32, activation = 'swish')(x)
            x = BatchNormalization()(x)
            x = Dense(16, activation = 'swish')(x)
            x = BatchNormalization()(x)
            x = Dense(nc , activation = 'softmax')(x)

            model  = Model(inputs=inpt, outputs = x)

            return model
