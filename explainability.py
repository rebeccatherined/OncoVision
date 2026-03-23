import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Generate Grad-CAM heatmap. Works for Functional API models.
    Automatically finds the last Conv2D layer.
    """
    # Find last Conv2D layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
        # Handle nested models (Sequential inside Functional)
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = sub_layer.name
                    # Build grad model on the sub-model
                    grad_model = tf.keras.models.Model(
                        inputs=layer.input,
                        outputs=[layer.get_layer(last_conv_layer_name).output, layer.output]
                    )
                    with tf.GradientTape() as tape:
                        conv_outputs, features = grad_model(img_array)
                        x = features
                        # Pass through remaining layers
                        found = False
                        for top_layer in model.layers:
                            if found:
                                x = top_layer(x)
                            if top_layer.name == layer.name:
                                found = True
                        class_channel = x[:, 0] if pred_index is None else x[:, pred_index]
                    grads = tape.gradient(class_channel, conv_outputs)
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    conv_outputs = conv_outputs[0]
                    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                    heatmap = tf.squeeze(heatmap)
                    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
                    return heatmap.numpy()

    if last_conv_layer_name is None:
        raise ValueError("No Conv2D layer found in model.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    if original_image.dtype != np.uint8:
        original_uint8 = np.uint8(original_image * 255)
    else:
        original_uint8 = original_image.copy()
    superimposed = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed