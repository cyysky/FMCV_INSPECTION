import tensorflow_hub as hub
import tensorflow as tf

from tensorflow.python.client import device_lib

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, Conv2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from keras import backend as K

from FMCV import Logging
import cv2
import os
import numpy as np
import time
import traceback
import json 
import matplotlib.pyplot as plt

from tkinter import simpledialog

from PIL import Image as pil_image

from pathlib import Path

np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

class CNNModel():
    def __init__(self, start, work_dir):
        self.start = start
        
        self.model_work_dir = work_dir
        self.model_full_path = None
        self.model_classes = None
        self.model = None
        self.model_gradient = None
        self.model_last_conv_layer = None
        
        self.model_datagen_kwargs = None
        
        self.image_shape = (224,224)
        
        
        self.load()
        
    def load(self):
        self.image_shape
        self.model_work_dir
        self.model_full_path
        self.model
        self.model_classes
        self.model_datagen_kwargs
        
        
    def save_model(self):  
        if self.start.Users.login_admin():
            # Save the weights
            try:
                os.makedirs(MODEL_PATH, exist_ok=True)
                
                model.save(MODEL)
                
                with open(os.path.join(MODEL_PATH,"name.json"), 'w') as f:
                    f.write(json.dumps(train_generator.class_indices))
                print("Model Saved")
            except:
                traceback.print_exc()
        
    
    def preheat(self):
        image_height = 224
        image_width = 224
        number_of_color_channels = 3
        color = (255,255,255)
        pixel_array = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
        self.predict("PreHeat",pixel_array)
    
    def VizGradCAM(self, model, image, interpolant=0.5, plot_results=False):
        gradient_model = self.model_gradient
        last_conv_layer = self.model_last_conv_layer
        
        """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
        using the gradients from the last convolutional layer. This function
        should work with all Keras Application listed here:
        https://keras.io/api/applications/
        Parameters:
        model (keras.model): Compiled Model with Weights Loaded
        image: Image to Perform Inference On
        plot_results (boolean): True - Function Plots using PLT
                                False - Returns Heatmap Array
        Returns:
        Heatmap Array?
        """
        #sanity check
        assert (interpolant > 0 and interpolant < 1), "Heatmap Interpolation Must Be Between 0 - 1"
        #STEP 1: Preprocesss image and make prediction using our model
        #input image
        original_img = np.asarray(image, dtype = np.float32)
        #expamd dimension and get batch size
        img = np.expand_dims(original_img, axis=0)
        #predict
        prediction = model.predict(img)
        #dict_class = train_generator.class_indices
        #dict_class = {v: k for k, v in dict_class.items()}
        #target_label= dict_class.get(np.argmax(label))
        #predicted_label = dict_class.get(np.argmax(prediction))
        classed = prediction[0]
        scored = round(float(np.max(prediction)),5)
        #print(target_label)
        #print(predicted_label)
        print(scored)
        #prediction index
        prediction_idx = np.argmax(prediction)
        #STEP 2: Create new model
        #specify last convolutional layer
        if last_conv_layer is None:
            last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
            target_layer = model.get_layer(last_conv_layer.name)
        #compute gradient of top predicted class
        with tf.GradientTape() as tape:
            #create a model with original model inputs and the last conv_layer as the output
            if gradient_model is None:
                gradient_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
            #pass the image through the base model and get the feature map  
            conv2d_out, prediction = gradient_model(img)
            #prediction loss
            loss = prediction[:, prediction_idx]
        #gradient() computes the gradient using operations recorded in context of this tape
        gradients = tape.gradient(loss, conv2d_out)
        #obtain the output from shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
        output = conv2d_out[0]
        #obtain depthwise mean
        weights = tf.reduce_mean(gradients[0], axis=(0, 1))
        #create a 7x7 map for aggregation
        activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
        #multiply weight for every layer
        for idx, weight in enumerate(weights):
            activation_map += weight * output[:, :, idx]
        #resize to image size
        activation_map = cv2.resize(activation_map.numpy(), 
                                    (original_img.shape[1], 
                                     original_img.shape[0]))
        #ensure no negative number
        activation_map = np.maximum(activation_map, 0)
        #convert class activation map to 0 - 255
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        #rescale and convert the type to int
        activation_map = np.uint8(255 * activation_map)
        #convert to heatmap
        heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
        #superimpose heatmap onto image
        original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
        cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended_heatmap = np.uint8(original_img * interpolant + heatmap * (1 - interpolant))
        #cv2.imwrite(f"{target_label}_{predicted_label}_{scored}.png",original_img * interpolant + heatmap * (1 - interpolant))
        #cvt_heatmap = img_to_array(cvt_heatmap)
        #enlarge plot
        #plt.rcParams["figure.dpi"] = 100

        #if plot_results == True:
        #    plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
        #else:
        #    return cvt_heatmap
            
        return np.argmax(classed), scored, blended_heatmap
    
def load():
    global start 

    global IMAGE_SHAPE #
    global MODEL_PATH # working directory
    global MODEL
    global model
    global dict_class
    global datagen_kwargs
    global base_model_path
    
    global learning_rate
    
    global gradient_model
    global last_conv_layer

    
    base_model_path = os.path.join("Profile",start.Config.profile)
    
    if str(start.Config.model_path) != ".":
        base_model_path = start.Config.model_path
        base_model_path.mkdir(parents=True, exist_ok=True)
        
    if start.Config.cnn_mode.casefold() == "normal":
        MODEL_PATH = os.path.join(base_model_path,"CNN_MODEL")
        MODEL = MODEL_PATH
        IMAGE_SHAPE = (224,224) 
        datagen_kwargs = dict(rescale=1./255)
        
    if start.Config.cnn_mode.casefold() in ("andrew_fast", "andrew"):
        MODEL_PATH = os.path.join(base_model_path,"ANDREW_CNN_MODEL")
        MODEL = MODEL_PATH
        IMAGE_SHAPE = (224,224)         
        datagen_kwargs = dict(rescale=1./255)
        
    if start.Config.cnn_mode.casefold() == "heatmap":
        MODEL_PATH = os.path.join(base_model_path,"CNN_MODEL")
        MODEL = os.path.join(MODEL_PATH,"advanced_trained.h5")
        IMAGE_SHAPE = (384,384)
        datagen_kwargs = {}
        
        os.makedirs(MODEL_PATH, exist_ok=True)
        
    if start.Config.cnn_mode.casefold() == "efficientnetv2xl":
        MODEL_PATH = os.path.join(base_model_path,"CNN_MODEL")
        MODEL = MODEL_PATH
        IMAGE_SHAPE = (512,512)
        datagen_kwargs = dict(rescale=1./255)
        
        
    if start.Config.cnn_mode.casefold() in ("normal","andrew_fast","heatmap","efficientnetv2xl"):    
        learning_rate = 0.001
        
    if start.Config.cnn_mode.casefold() ==("andrew"):
        learning_rate = 0.0001
        
    Logging.info(device_lib.list_local_devices())
    Logging.info(MODEL)
    Logging.info(MODEL_PATH)
    Logging.info(IMAGE_SHAPE)

    if gradient_model is not None:
        gradient_model = None
        last_conv_layer = None
        target_layer = None
        
    if model is not None:
        model = None
        tf.keras.backend.clear_session()
        Logging.info("Clear loaded model")
    
    if os.path.exists(MODEL):
        model = tf.keras.models.load_model(MODEL)
        #model.summary()
        try:                
            with open(os.path.join(MODEL_PATH,"name.json")) as f:
                dict_class = json.loads(f.read())
        except:
            traceback.print_exc()
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
             loss=tf.keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])
    else:
        dict_class = None
        #Logging.info(model.optimizer.get_config())
    Logging.info(learning_rate)
        
def get_class_name(n):
    global dict_class    
    if dict_class is None:
        return str(n)
    else:
        try:
            key_list=list(dict_class.keys())
            val_list=list(dict_class.values())
            ind=val_list.index(n)
            return key_list[ind]
        except:
            traceback.print_exc()
            return str(n)

def predict(name,im): #im is OpenCv BGR Numpy NdArray format
    ''' 
        predict inference CNN with selected model in config
        ai_model == "MobileNetV3"
        ai_model == "heatmap"
        name : name of the prediction
        im : OpenCv BGR Numpy NdArray format
    '''
    global start
    
    global model
    global IMAGE_SHAPE
    global datagen_kwargs
    
    if model is not None:       
        
        p = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        p = pil_image.fromarray(p)
        p = p.resize(IMAGE_SHAPE,pil_image.Resampling.NEAREST)
        p = np.asarray(p, dtype='float32')

        if start.Config.cnn_mode.casefold() in ("normal","andrew_fast","andrew","efficientnetv2xl"):
            scaled_im = p * datagen_kwargs['rescale'] 
        elif start.Config.cnn_mode.casefold() in ("heatmap"):
            scaled_im = p
        
        Logging.info(f"s={scaled_im[0][0][0]}")
        
        start_time = time.time()
        if start.Config.cnn_mode.casefold() == "heatmap":
            return VizGradCAM(model, scaled_im)
        else:
            input_im = np.expand_dims(scaled_im, axis=0)
            output = model.predict(input_im)
            score = output[0]
            print(f'{name} {np.argmax(score)} ,{round(float(np.max(score)),5)} ,{score} time {round(time.time() - start_time,5)}')
            #print(output)
            return np.argmax(score),round(float(np.max(score)),5),None

    return 0,0,None
        
def write_images(dir,sub,template):
    w,h = template.shape[::-1]
    for file in os.listdir(dir):
        if file.endswith(".jpg"):
            print(file)
            if not os.path.exists(os.path.join("TRAIN",sub,file)):
                temp_im_color = cv2.imread(os.path.join(dir,file))
                temp_im=cv2.cvtColor(temp_im_color, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(temp_im,template,cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                x1,y1 = max_loc            
                cropped_im = temp_im_color[y1:y1+h,x1:x1+w]            
                cv2.imwrite(os.path.join("TRAIN",sub,file),cropped_im)   

def create_model(classes):
    global model
    
    global learning_rate
    
    model = None
    
    tf.keras.backend.clear_session()
    
    if start.Config.cnn_mode.casefold() == "normal":
        model = tf.keras.Sequential([
         hub.KerasLayer(os.path.join("FMCV","Ai","AI"),trainable=False),
         #tf.keras.layers.GlobalAveragePooling2D(),
         #tf.keras.layers.Dense(1280, activation='sigmoid'),  
         #tf.keras.layers.Dropout(.2),
         tf.keras.layers.Dense(classes, activation='softmax')
        ])
        model.build([None, 224, 224, 3])
         
    if start.Config.cnn_mode.casefold() in ("andrew_fast","andrew"):
        model = tf.keras.Sequential([
         hub.KerasLayer(os.path.join("FMCV","Ai","AI"),trainable=False),
         #tf.keras.layers.GlobalAveragePooling2D(),
         #tf.keras.layers.Dropout(.2)
         #tf.keras.layers.Dense(960, activation='relu'),
         tf.keras.layers.Dense(640, activation='relu'),  
         #tf.keras.layers.Dropout(.2),
         tf.keras.layers.Dense(classes, activation='softmax')
        ])
        model.build([None, 224, 224, 3])

         
    if start.Config.cnn_mode.casefold() == "heatmap":
    
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=(384, 384, 3)
        )

        base_model.trainable = False

        #build the model
        model = base_model.output
        model = tf.keras.layers.GlobalAveragePooling2D()(model) # this is needed for EfficientNet
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Dense(640, activation='relu')(model)
        model = tf.keras.layers.Dense(classes, activation='softmax')(model)
        
        model = tf.keras.models.Model(inputs=base_model.input, outputs=model)

    
    if start.Config.cnn_mode.casefold() == "efficientnetv2xl":
    
        model = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
                           trainable=False),  # Can be True, see below.
            tf.keras.layers.Dense(classes, activation='softmax')
        ])
        model.build([None, 512, 512, 3])  # Batch input shape.
     
    if model is not None:
        model.compile(
         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),#.experimental.RMSprop(learning_rate=1e-3),#optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=1e-3),
         loss=tf.keras.losses.CategoricalCrossentropy(),
         metrics=['accuracy'])
            
def train(epochs = 0): 
    if start.Users.login_admin():
        global MODEL 
        #MODEL = os.path.join("Profile",start.Config.profile,"CNN_MODEL")
        
        global model
        global dict_class
        global datagen_kwargs
        
        global IMAGE_SHAPE
        
        global train_generator
        
        
        batch_size = 32
        
        if start.Config.cnn_mode.casefold() in ("andrew_fast","andrew"):
            batch_size = 16
        
        data_root = os.path.join(base_model_path,"images")
        
        TRAINING_DATA_DIR = str(data_root)
        
        
        datagen_kwargs.update(rotation_range=start.Config.train_rotate)
            
        if start.Config.train_brightness > 0:            
            datagen_kwargs.update(brightness_range=(1-start.Config.train_brightness, 1+start.Config.train_brightness))
        
        if start.Config.train_width_shift > 0:            
            datagen_kwargs.update(width_shift_range=start.Config.train_width_shift)
            
        if start.Config.train_height_shift > 0:            
            datagen_kwargs.update(height_shift_range=start.Config.train_height_shift)
            
        if start.Config.train_zoom_range > 0:            
            datagen_kwargs.update(zoom_range=start.Config.train_zoom_range)
            
        if start.Config.train_horizontal_flip:            
            datagen_kwargs.update(horizontal_flip=start.Config.train_horizontal_flip)
            
        if start.Config.train_vertical_flip:            
            datagen_kwargs.update(vertical_flip=start.Config.train_vertical_flip)
        
        Logging.info(datagen_kwargs)
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
        
        train_generator_kwargs = {}
        train_generator_kwargs.update(shuffle=True,
                                        target_size=IMAGE_SHAPE,
                                        batch_size = batch_size)
        if start.Config.train_save_augmentation:
            save_dir = Path(base_model_path,'augmenation_output')
            save_dir.mkdir(parents=True, exist_ok=True)
            train_generator_kwargs.update(save_to_dir=save_dir,save_prefix='aug',save_format='png')
        
        Logging.debug(train_generator_kwargs)
        
        train_generator = train_datagen.flow_from_directory(TRAINING_DATA_DIR,**train_generator_kwargs)

        dict_class = train_generator.class_indices
        print(train_generator.class_indices) 
        #https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5
        #https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5
        #https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4
        
        if model is None:
            create_model(train_generator.num_classes)
        
        if epochs == 0:
            epochs = int(simpledialog.askstring(title="CNN Training", prompt="Epochs Number :"))
            #epochs = int(input("Epochs? : "))
        
        fit_callbacks = []
        
        #define checkpoint #not using
        checkpoint = ModelCheckpoint(MODEL, 
                                     monitor='loss',
                                     save_best_only=True,
                                     mode='min',
                                     verbose=1)

        #early stopping #not using
        earlystop = EarlyStopping(monitor='accuracy',
                                  patience=5,
                                  mode='auto',
                                  verbose=1)

        #reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor = 'accuracy', 
                                      factor = 0.1, 
                                      patience = 10, 
                                      min_delta = 0.0001,
                                      mode='auto',
                                      verbose=1)
        if start.Config.cnn_auto_save:
            fit_callbacks = [checkpoint]# reduce_lr,earlystop not using
        else:
            fit_callbacks = []
        
        if start.Config.debug_level < 40:
            verbose = 1
        else:
            verbose = 1
        
        try:
            steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
            
            hist = model.fit(
             train_generator,
             epochs=epochs,
             verbose=verbose,
             steps_per_epoch=steps_per_epoch,
             batch_size = train_generator.batch_size,
             callbacks=fit_callbacks
             ).history
        except KeyboardInterrupt:
            print("\nManual Ended, Saving CNN Model")
        except:    
            traceback.print_exc()
            create_model(train_generator.num_classes)
            steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
            hist = model.fit(
             train_generator,
             epochs=epochs,
             verbose=verbose,
             batch_size = train_generator.batch_size,
             steps_per_epoch=steps_per_epoch,
             callbacks=fit_callbacks
             ).history

        save_model()
    

        


