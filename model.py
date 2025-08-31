from tensorflow import keras
from tensorflow.keras import layers

class ChangeDetectionModel:
    @staticmethod
    def conv_block(x, filters, kernel_size=3, activation='relu'):
        x=layers.Conv2D(filters,kernel_size,padding='same')(x)
        x=layers.BatchNormalization()(x)
        x=layers.Activation(activation)(x)
        return x

    @staticmethod
    def encoder_block(x,filters):
        x=ChangeDetectionModel.conv_block(x,filters)
        x=ChangeDetectionModel.conv_block(x,filters)
        skip=x
        x=layers.MaxPooling2D(2)(x)
        return x,skip

    @staticmethod
    def decoder_block(x,skip,filters):
        x=layers.UpSampling2D(2)(x)
        x=layers.Concatenate()([x,skip])
        x=ChangeDetectionModel.conv_block(x,filters)
        x=ChangeDetectionModel.conv_block(x,filters)
        return x

    @staticmethod
    def build_siamese_unet(config):
        input1=layers.Input(shape=(config.TILE_SIZE,config.TILE_SIZE,config.INPUT_CHANNELS))
        input2=layers.Input(shape=(config.TILE_SIZE,config.TILE_SIZE,config.INPUT_CHANNELS))
        def create_encoder(inp):
            x,s1=ChangeDetectionModel.encoder_block(inp,64)
            x,s2=ChangeDetectionModel.encoder_block(x,128)
            x,s3=ChangeDetectionModel.encoder_block(x,256)
            x,s4=ChangeDetectionModel.encoder_block(x,512)
            x=ChangeDetectionModel.conv_block(x,1024)
            x=ChangeDetectionModel.conv_block(x,1024)
            return x,[s1,s2,s3,s4]
        enc1,sk1=create_encoder(input1)
        enc2,sk2=create_encoder(input2)
        diff=layers.Subtract()([enc1,enc2])
        x=ChangeDetectionModel.decoder_block(diff,layers.Concatenate()([sk1[3],sk2[3]]),512)
        x=ChangeDetectionModel.decoder_block(x,layers.Concatenate()([sk1[2],sk2[2]]),256)
        x=ChangeDetectionModel.decoder_block(x,layers.Concatenate()([sk1[1],sk2[1]]),128)
        x=ChangeDetectionModel.decoder_block(x,layers.Concatenate()([sk1[0],sk2[0]]),64)
        output=layers.Conv2D(1,1,activation='sigmoid',padding='same')(x)
        return keras.Model(inputs=[input1,input2],outputs=output,name="SiameseUNet")
