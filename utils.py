import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16, DenseNet121, MobileNetV2, InceptionV3
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout,
    Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Multiply, Concatenate, Reshape, SeparableConv2D, Add
)
from tensorflow.keras.models import Model
import numpy as np
import json
import pickle

IMG_SIZE = (224, 224)


# ─────────────────────────────────────────────────────────────────
# CUSTOM MAMMOGRAM CNN — exact match to training notebook
# ─────────────────────────────────────────────────────────────────
def _residual_block(x, filters, stride=1):
    shortcut = x
    x = SeparableConv2D(filters, 3, padding='same', strides=stride, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def _build_custom_mammo_cnn():
    inp = Input(shape=(224, 224, 3))
    x = Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x = _residual_block(x, 32)
    x = _residual_block(x, 64,  stride=2)
    x = _residual_block(x, 128, stride=2)
    x = _residual_block(x, 256, stride=2)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='Custom_ResNet_CNN')


# ─────────────────────────────────────────────────────────────────
# CUSTOM ULTRASOUND CNN — exact match to training notebook
# ─────────────────────────────────────────────────────────────────
def _channel_attention(x, ratio=8):
    c = x.shape[-1]
    avg = GlobalAveragePooling2D()(x)
    avg = Reshape((1, 1, c))(avg)
    avg = Dense(c // ratio, activation='relu',    use_bias=False)(avg)
    avg = Dense(c,          activation='sigmoid', use_bias=False)(avg)
    return Multiply()([x, avg])


def _conv_bn_relu(x, filters, stride=1):
    x = Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def _build_custom_us_cnn():
    inp = Input(shape=(224, 224, 3))
    x = _conv_bn_relu(inp, 32, stride=2)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x = _conv_bn_relu(x, 64);  x = _conv_bn_relu(x, 64)
    x = _channel_attention(x); x = MaxPooling2D(2, 2)(x)
    x = _conv_bn_relu(x, 128); x = _conv_bn_relu(x, 128)
    x = _channel_attention(x); x = MaxPooling2D(2, 2)(x)
    x = _conv_bn_relu(x, 256); x = _conv_bn_relu(x, 256)
    x = _channel_attention(x); x = MaxPooling2D(2, 2)(x)
    avg = GlobalAveragePooling2D()(x)
    mx  = GlobalMaxPooling2D()(x)
    x = Concatenate()([avg, mx])
    x = Dense(256, activation='relu')(x); x = Dropout(0.4)(x)
    x = Dense(64,  activation='relu')(x); x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='Custom_Attention_CNN')


# ─────────────────────────────────────────────────────────────────
# MAMMOGRAM MODELS — exactly as built in MamogramModels_final.ipynb
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # 1. Custom CNN
    model1 = _build_custom_mammo_cnn()
    model1.load_weights("models/model_custom.weights.h5")

    # 2. VGG16 Feature — base(inp, training=False) pattern
    base_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_vgg.trainable = False
    inp_vgg = Input(shape=(224, 224, 3))
    x = base_vgg(inp_vgg, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    model2 = Model(inp_vgg, Dense(1, activation='sigmoid')(x), name='VGG16_Feature')
    model2.load_weights("models/model_vgg_feature.weights.h5")

    # 3. VGG16 FineTune — last 4 layers unfrozen, same base(inp, training=False) pattern
    base_vgg_ft = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    for layer in base_vgg_ft.layers[:-4]:
        layer.trainable = False
    for layer in base_vgg_ft.layers[-4:]:
        layer.trainable = True
    inp_vgg_ft = Input(shape=(224, 224, 3))
    x2 = base_vgg_ft(inp_vgg_ft, training=False)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    model3 = Model(inp_vgg_ft, Dense(1, activation='sigmoid')(x2), name='VGG16_FineTune')
    #model3.load_weights("models/model_vgg_finetune.weights.h5")
    model3.load_weights("models/model_vgg_finetune.weights.h5", skip_mismatch=True)

    # 4. DenseNet121 — base(inp, training=False) pattern
    base_dense = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_dense.trainable = False
    inp_dense = Input(shape=(224, 224, 3))
    x3 = base_dense(inp_dense, training=False)
    x3 = GlobalAveragePooling2D()(x3)
    x3 = Dropout(0.5)(x3)
    model4 = Model(inp_dense, Dense(1, activation='sigmoid')(x3), name='DenseNet121')
    model4.load_weights("models/model_densenet.weights.h5")

    # 5. MobileNetV2 — base(inp, training=False) pattern
    base_mob = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_mob.trainable = False
    inp_mob = Input(shape=(224, 224, 3))
    x4 = base_mob(inp_mob, training=False)
    x4 = GlobalAveragePooling2D()(x4)
    x4 = Dropout(0.5)(x4)
    model5 = Model(inp_mob, Dense(1, activation='sigmoid')(x4), name='MobileNetV2')
    model5.load_weights("models/model_mobilenet.weights.h5")

    return {
        "Custom CNN":     model1,
        "VGG16 Feature":  model2,
        "VGG16 FineTune": model3,
        "DenseNet121":    model4,
        "MobileNetV2":    model5,
    }


# ─────────────────────────────────────────────────────────────────
# ULTRASOUND MODELS — exactly as built in UltrasoundModels_v3_final.ipynb
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ultrasound_models():
    # 1. Custom Attention CNN
    model_custom = _build_custom_us_cnn()
    model_custom.load_weights("models/model_ultrasound_custom.weights.h5")

    # 2. DenseNet121
    base_dense = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_dense.trainable = False
    inp_dense = Input(shape=(224, 224, 3))
    x = base_dense(inp_dense, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    model_dense = Model(inp_dense, Dense(1, activation='sigmoid')(x), name='DenseNet121_US')
    model_dense.load_weights("models/model_ultrasound_densenet.weights.h5")

    # 3. MobileNetV2
    base_mob = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_mob.trainable = False
    inp_mob = Input(shape=(224, 224, 3))
    x2 = base_mob(inp_mob, training=False)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dropout(0.5)(x2)
    model_mob = Model(inp_mob, Dense(1, activation='sigmoid')(x2), name='MobileNetV2_US')
    model_mob.load_weights("models/model_ultrasound_mobilenet.weights.h5")

    # 4. InceptionV3 — 299x299
    base_inc = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    base_inc.trainable = False
    inp_inc = Input(shape=(299, 299, 3))
    x3 = base_inc(inp_inc, training=False)
    x3 = GlobalAveragePooling2D()(x3)
    x3 = Dropout(0.5)(x3)
    model_inc = Model(inp_inc, Dense(1, activation='sigmoid')(x3), name='InceptionV3_US')
    model_inc.load_weights("models/model_ultrasound_inceptionv3.weights.h5")

    return {
        "Custom CNN":  model_custom,
        "DenseNet121": model_dense,
        "MobileNetV2": model_mob,
        "InceptionV3": model_inc,
    }


# ─────────────────────────────────────────────────────────────────
# DENSITY MODEL — exactly as built in BreastDensity_v7_final.ipynb
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_density_model():
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base.trainable = False
    inp = Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out, name='MobileNetV2_Density')
    model.load_weights("models/model_density.weights.h5")
    return model


@st.cache_resource
def load_density_meta():
    with open("density_meta.pkl", "rb") as f:
        return pickle.load(f)


def predict_density(model, img, meta):
    pred       = float(model.predict(img, verbose=0).squeeze())
    threshold  = meta.get('threshold', 0.39)
    dense_idx  = meta.get('dense_idx', 1)
    prob_dense = pred if dense_idx == 1 else (1.0 - pred)
    is_dense   = prob_dense >= threshold
    label      = "Dense" if is_dense else "Non-Dense"
    return is_dense, prob_dense, label


# ─────────────────────────────────────────────────────────────────
# META / ENSEMBLE MODELS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_meta_model():
    with open("ensemble_meta.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_ultrasound_meta():
    with open("ultrasound_meta.pkl", "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data
    return {'model': data, 'threshold': 0.5, 'model_thresholds': {}}


# ─────────────────────────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────────────────────────
def load_classes():
    with open("class_indices_mammogram.json") as f:
        data = json.load(f)
    return {v: k for k, v in data.items()}


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────
def preprocess_image(pil_image, size=(224, 224)):
    img = pil_image.convert("RGB").resize(size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────
# STACKING ENSEMBLE PREDICT
# ─────────────────────────────────────────────────────────────────
def stacking_predict(models_dict, meta_data, img_224, pil_image=None):
    if isinstance(meta_data, dict):
        meta_model = meta_data['model']
        threshold  = meta_data.get('threshold', 0.5)
    else:
        meta_model = meta_data
        threshold  = 0.5

    preds = []
    for name, m in models_dict.items():
        if name == "InceptionV3" and pil_image is not None:
            inp299 = preprocess_image(pil_image, size=(299, 299))
            preds.append(m.predict(inp299, verbose=0))
        else:
            preds.append(m.predict(img_224, verbose=0))

    X    = np.hstack(preds)
    prob = float(meta_model.predict_proba(X)[:, 1][0])
    return prob, threshold


# ─────────────────────────────────────────────────────────────────
# MULTIMODAL FUSION
# ─────────────────────────────────────────────────────────────────
def fusion_predict(mammo_prob, ultra_prob, mammo_weight=0.6, ultra_weight=0.4):
    return mammo_weight * mammo_prob + ultra_weight * ultra_prob


# ─────────────────────────────────────────────────────────────────
# RISK INTERPRETATION
# ─────────────────────────────────────────────────────────────────
def interpret_risk(prob, threshold=0.5):
    if prob < threshold * 0.6:
        return "Low Risk", "green", "✅"
    elif prob < threshold:
        return "Moderate Risk", "orange", "⚠️"
    else:
        return "High Risk", "red", "🚨"