
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop
from keras.layers.advanced_activations import LeakyReLU
import time

import cv2
# warnings.filterwarnings('ignore')


# In[2]:

import os
path = os.path.join(os.getcwd(),'sketch-data')
image = os.path.join(path,'image')
_input = os.path.join(path,'t_input')
output = os.path.join(path,'t_output')
edges = os.path.join(path,'edge')
print 'import'

# In[3]:

class DCGAN(object):
    def __init__(self, img_rows=128, img_cols=128, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 128 x 128 x 1, depth = 1
        # Out: 64 x 64 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        
        self.D.add(Conv2D(depth*16, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
#         self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        _id = 64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.G.add(Conv2D(_id, 5, strides=1, input_shape=input_shape,            padding='same'))
        self.G.add(LeakyReLU(alpha=0.2))
        self.G.add(Dropout(dropout))
        #128X128X64

        self.G.add(Conv2D(_id*2, 5, strides=2, padding='same'))
        self.G.add(LeakyReLU(alpha=0.2))
        self.G.add(Dropout(dropout))
        #64X64X128
        
        
        self.G.add(Conv2D(_id*4, 5, strides=2, padding='same'))
        self.G.add(LeakyReLU(alpha=0.2))
        self.G.add(Dropout(dropout))
        #32X32X256
#         self.D.add(Conv2D(_id*4, 5, strides=1, padding='same'))
#         self.D.add(LeakyReLU(alpha=0.2))
#         self.D.add(Dropout(dropout))
#         #32X32X256
#         self.G.add(Dense(dim*dim*depth, input_dim=100))
#         self.G.add(BatchNormalization(momentum=0.9))
#         self.G.add(Activation('relu'))
#         self.G.add(Reshape((dim, dim, depth)))
#         self.G.add(Dropout(dropout))

#         # In: dim x dim x depth
#         # Out: 2*dim x 2*dim x depth/2
#         self.G.add(UpSampling2D())
#         self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
#         self.G.add(BatchNormalization(momentum=0.9))
#         self.G.add(Activation('relu'))
#         # 32X32X128
        
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        #64X64X64
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        #128X128X32
        
        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
#         self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,            metrics=['accuracy'])
        return self.AM


# In[4]:

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


# In[7]:

class SKETCH_CGAN(object):
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channel = 1
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         self.x_train = x_train[:len(x_train)/10]
#         import numpy as np
        #x_train = []
        #for img in os.listdir(_input):
        #    x_train+= [cv2.imread(os.path.join(_input,img),0)]
        #self.x_train = np.array(x_train)
#         x_train.shape
        #self.x_train = self.x_train.reshape(-1, self.img_rows,        	self.img_cols, 1).astype(np.float32)
	print 'init'
        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()
	print 'init'


    def read(self,li,j,batch_size,direc):
	mat = []
	for i in range(j*batch_size,(j+1)*batch_size):
		
		img = li[i]
		mat.append(cv2.imread(os.path.join(direc,img),0))
	mat = np.array(mat)
	mat = mat.reshape(-1, mat.shape[1],mat.shape[2], 1).astype(np.float32)
	print mat.shape
	return mat
	
    def train(self, train_steps=200, batch_size=256, save_interval=0):
#         noise_input = None
#         if save_interval>0:
#             noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        mat = []
	edge_list = list(os.listdir(output))	
        #for img in os.listdir(output):
                #mat.append(cv2.imread(os.path.join(output,img),0))
		#print len(mat),
        #mat = np.array(mat)
        #mat = mat.reshape(-1, self.img_rows,                self.img_cols, 1).astype(np.float32)
        real = []
	print 'real'
        real_list = list(os.listdir(_input))
	#for img in os.listdir(_input):
                #real.append(cv2.imread(os.path.join(_input,img),0))
		#print len(real),
        #real = np.array(real)
        #real = real.reshape(-1, self.img_rows,                self.img_cols, 1).astype(np.float32)
        #batch_size = mat.shape[0]
        for i in range(train_steps):
            
#                 images_train = self.x_train[np.random.randint(0,
#                     self.x_train.shape[0], size=batch_size), :, :, :]
#                 noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
          n=50
	  for j in range(n-1):   
	    print i,j
	    batch_size = int(len(real_list)/n)	
	    _in = self.read(edge_list,j,batch_size,output)#mat[j*batch_size: (j+1)*batch_size]
	    _out = self.read(real_list,j,batch_size,_input)#real[j*batch_size:(j+1)*batch_size]
            images_fake = self.generator.predict(_in)
            #print mat.shape, images_fake.shape
	    print _in.shape, _out.shape	
            x = np.concatenate((_out, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
#             noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(_in, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
#             if save_interval>0:
#                 if (i+1)%save_interval==0:
#                     self.plot_images(save2file=True, samples=noise_input.shape[0],\
#                         noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


# In[8]:

mnist_dcgan = SKETCH_CGAN()
timer = ElapsedTimer()
mnist_dcgan.train(train_steps=1000, batch_size=256, save_interval=500)
timer.elapsed_time()


# In[ ]:

mnist_dcgan.generator.save('gen.pkl')
mnist_dcgan.adversarial.save('ad.pkl')
mnist_dcgan.discriminator.save('dis.pkl')
#mnist_dcgan.plot_images(fake=True)
#mnist_dcgan.plot_images(fake=False)


# In[ ]:



