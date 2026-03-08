import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, DenseNet121, MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, Model
import numpy as np
import json
import pickle

IMG_SIZE = (224,224)

# ---------------------------------------------------
# MAMMOGRAM MODELS
# ---------------------------------------------------
@st.cache_resource
def load_models():

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

    inputs = Input(shape=(224,224,3))
    base_vgg = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    x = GlobalAveragePooling2D()(base_vgg.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    vgg_feature_model = Model(inputs, outputs)
    vgg_feature_model.load_weights("models/model_vgg_feature.weights.h5")

    inputs2 = Input(shape=(224,224,3))
    base_vgg_ft = VGG16(include_top=False, weights="imagenet", input_tensor=inputs2)
    x2 = GlobalAveragePooling2D()(base_vgg_ft.output)
    x2 = Dropout(0.5)(x2)
    outputs2 = Dense(1, activation='sigmoid')(x2)
    vgg_finetune_model = Model(inputs2, outputs2)
    vgg_finetune_model.load_weights("models/model_vgg_finetune.weights.h5")

    base_res = ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
    resnet_model = Sequential([
        base_res,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    resnet_model.load_weights("models/model_resnet.weights.h5")

    base_eff = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
    efficient_model = Sequential([
        base_eff,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    efficient_model.load_weights("models/model_efficientnet.weights.h5")

    base_dense = DenseNet121(include_top=False, weights="imagenet", input_shape=(224,224,3))
    densenet_model = Sequential([
        base_dense,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    densenet_model.load_weights("models/model_densenet.weights.h5")

    base_mobile = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
    mobilenet_model = Sequential([
        base_mobile,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    mobilenet_model.load_weights("models/model_mobilenet.weights.h5")

    return {
        "Custom CNN": custom_model,
        "VGG16 Feature": vgg_feature_model,
        "VGG16 FineTune": vgg_finetune_model,
        "ResNet50": resnet_model,
        "EfficientNetB0": efficient_model,
        "DenseNet121": densenet_model,
        "MobileNetV2": mobilenet_model
    }


# ---------------------------------------------------
# ULTRASOUND MODELS
# ---------------------------------------------------
@st.cache_resource
def load_ultrasound_models():

    eff = Sequential([
        EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3)),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1,activation="sigmoid")
    ])
    eff.load_weights("models/model_ultrasound_efficientnet.weights.h5")

    dense = Sequential([
        DenseNet121(include_top=False, weights="imagenet", input_shape=(224,224,3)),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1,activation="sigmoid")
    ])
    dense.load_weights("models/model_ultrasound_densenet.weights.h5")

    mobile = Sequential([
        MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3)),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1,activation="sigmoid")
    ])
    mobile.load_weights("models/model_ultrasound_mobilenet.weights.h5")

    return {
        "EfficientNet": eff,
        "DenseNet": dense,
        "MobileNet": mobile
    }


# ---------------------------------------------------
# DENSITY MODEL
# ---------------------------------------------------
@st.cache_resource
def load_density_model():

    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(4, activation="softmax")(x)

    model = Model(base.input, output)
    model.load_weights("models/model_density.weights.h5")

    return model


def predict_density(model,img):
    pred = model.predict(img)
    return int(np.argmax(pred))


# ---------------------------------------------------
# META MODELS
# ---------------------------------------------------
@st.cache_resource
def load_meta_model():
    with open("ensemble_meta.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_ultrasound_meta():
    with open("ultrasound_meta.pkl","rb") as f:
        return pickle.load(f)


# ---------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------
def load_classes():
    with open("class_indices_mammogram.json") as f:
        data=json.load(f)
    return dict((v,k) for k,v in data.items())


def load_density_classes():
    with open("class_indices_density.json") as f:
        data=json.load(f)
    return dict((v,k) for k,v in data.items())


# ---------------------------------------------------
# PREPROCESS
# ---------------------------------------------------
def preprocess_image(image):
    image=image.convert("RGB")
    image=image.resize((224,224))
    image=np.array(image)/255.0
    image=np.expand_dims(image,axis=0)
    return image


# ---------------------------------------------------
# STACKING
# ---------------------------------------------------
def stacking_predict(models,meta_model,img):

    preds=[]
    for m in models.values():
        preds.append(m.predict(img))

    X=np.hstack(preds)
    return meta_model.predict_proba(X)[:,1]


def ultrasound_stacking_predict(models,meta_model,img):

    preds=[]
    for m in models.values():
        preds.append(m.predict(img))

    X=np.hstack(preds)
    return meta_model.predict_proba(X)[:,1]


# ---------------------------------------------------
# FUSION (Weighted)
# ---------------------------------------------------
def fusion_predict(mammo_prob, ultra_prob):

    fusion_prob = 0.6 * mammo_prob + 0.4 * ultra_prob

    return fusion_prob