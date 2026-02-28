import numpy as np
import tensorflow as tf
import cv2

# ----------------------------------------------------
# Automatically find last convolutional layer
# ----------------------------------------------------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


# ----------------------------------------------------
# Grad-CAM Heatmap Generator
# ----------------------------------------------------
def make_gradcam_heatmap(img_array, model):

    last_conv_layer_name = get_last_conv_layer(model)

    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found in model.")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()


# ----------------------------------------------------
# Overlay Heatmap on Original Image
# ----------------------------------------------------
def overlay_heatmap(heatmap, image, alpha=0.4):

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + image

    return superimposed_img.astype("uint8")