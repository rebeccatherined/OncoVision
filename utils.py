import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, Model
import numpy as np
import json
import pickle

IMG_SIZE = (224,224)

@st.cache_resource
def load_models():

    # Custom CNN
    custom_model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(1,activation='sigmoid')
    ])
    custom_model.load_weights("models/model_custom.weights.h5")

    # VGG Feature
    inputs = Input(shape=(224,224,3))
    base_vgg = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    base_vgg.trainable = False
    x = base_vgg.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    vgg_feature_model = Model(inputs, outputs)
    vgg_feature_model.load_weights("models/model_vgg_feature.weights.h5")

    # VGG Finetune
    inputs2 = Input(shape=(224,224,3))
    base_vgg_ft = VGG16(include_top=False, weights="imagenet", input_tensor=inputs2)
    base_vgg_ft.trainable = True
    x2 = base_vgg_ft.output
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dropout(0.5)(x2)
    outputs2 = Dense(1, activation='sigmoid')(x2)
    vgg_finetune_model = Model(inputs2, outputs2)
    vgg_finetune_model.load_weights("models/model_vgg_finetune.weights.h5")

    # ResNet50
    
    base_res = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224,224,3)
    )
    base_res.trainable = False

    resnet_model = Sequential([
        base_res,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    resnet_model.load_weights("models/model_resnet.weights.h5")
    

    return {
        "Custom CNN": custom_model,
        "VGG16 Feature": vgg_feature_model,
        "VGG16 FineTune": vgg_finetune_model,
        "ResNet50": resnet_model
    }

@st.cache_resource
def load_meta_model():
    with open("ensemble_meta.pkl", "rb") as f:
        meta_model = pickle.load(f)
    return meta_model

def load_classes():
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    return dict((v,k) for k,v in class_indices.items())

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

def stacking_predict(models, meta_model, img):
    preds = []
    for model in models.values():
        pred = model.predict(img)
        preds.append(pred)

    X_meta = np.hstack(preds)
    final_prob = meta_model.predict_proba(X_meta)[:,1]

    return final_prob