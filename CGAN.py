from __future__ import print_function, division
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Embedding
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np

class CGAN():
    def __init__(self,L_num:int,F_num:int,channels_num:int,num_classes:int,latent_dim=100):
        """
        Initialize the CGAN model.
        
        Args:
            self (CGAN): The CGAN model.
            L_num (int): L dimension number.
            F_num (int): F dimension number.
            channels_num (int): Feature channel number.
            num_classes (int): Class number.
            latent_dimension (int): Number of latent dimension (default as 100).
        """
        
        # Input shape
        self.L_num = L_num
        self.F_num = F_num
        self.channels_num = channels_num
        self.feature_shape = (self.L_num, self.F_num, self.channels_num)
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Set the optimizer
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        feature = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.generator.trainable = True

        # The discriminator takes generated features as input and determines validity
        # and the label of that features
        valid = self.discriminator([feature, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=[
           'binary_crossentropy'],
            optimizer=optimizer)
    
    
    def update_model(self,g_path,d_path,c_path):
        """
        Load existing trained model.
        
        Args:
            self (CGAN): The CGAN model.
            g_path (str): Path to load G
            d_path (str): Path to load D
            c_path (str): Path to load combined model
        """
        d = tf.keras.models.load_model(d_path) # Load discriminator
        g = tf.keras.models.load_model(g_path) # Load generator
        c = tf.keras.models.load_model(c_path) # Load conbined model
        return d,g,c

    def build_generator(self):
        """
        Build the generator model.
        
        Args:
            self (CGAN): The CGAN model.
        """
        # Creating a sequential model
        model = Sequential()
        
        # Layer1: a dense layer with 256 units and input dimension as self.latent_dim
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # Layer2: a dense layer with 512 units
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # Layer3: a dense layer with 1024 units
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # Layer4: a dense layer using 'tanh' activation function
        model.add(Dense(np.prod(self.feature_shape), activation='tanh'))
        model.add(Reshape(self.feature_shape))
        
        # Printing the summary of the model
        model.summary()

        noise = Input(shape=(self.latent_dim,)) # Input tensor for the noise
        label = Input(shape=(1,), dtype='int32')  # Input tensor for the label with dtype 'int32'
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label)) # Embedding the labels and flattening the output

        model_input = multiply([noise, label_embedding])  # Element-wise multiplication of the noise and label_embedding
        feature = model(model_input) # Passing the model_input through the generator model to get the generated feature

        return Model([noise, label], feature)

    
    def build_discriminator(self):
        """
        Build the discriminator model.
        
        Args:
            self (CGAN): The CGAN model.
        """
        # Creating a sequential model
        model = Sequential()  
        
        # Layer1: dense layer with 512 units
        model.add(Dense(512, input_dim=np.prod(self.feature_shape)))  
        model.add(LeakyReLU(alpha=0.2))  # Adding LeakyReLU activation with a negative slope of 0.2
        
        # Layer2: a dense layer with 512 units
        model.add(Dense(512))  
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dropout(0.4)) 
    
        # Layer3:  a dense layer with 512 units
        model.add(Dense(512))  
        model.add(LeakyReLU(alpha=0.2))  
        model.add(Dropout(0.4))

        # Layer4: a dense layer with 1 unit and 'sigmoid' function
        model.add(Dense(1, activation='sigmoid'))  
        model.summary()  # Printing the summary of the model
    
        feature = Input(shape=self.feature_shape)  # input tensor for the feature
        label = Input(shape=(1,), dtype='int32')  # input tensor for the label
    
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.feature_shape))(label))  # Embedding the labels and flattening the output
        flat_feature = Flatten()(feature)  # Flattening the feature te nsor
    
        model_input = multiply([flat_feature, label_embedding])  # Element-wise multiplication of flat_feature and label_embedding
        validity = model(model_input)  # Passing the model_input through the discriminator model to get the validity
    
        return Model([feature, label], validity) 
