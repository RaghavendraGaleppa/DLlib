"""
    This module will have functions related to interpretability of 
    a model.
"""
import tensorflow.keras.backend as K
import cv2
import matplotlib.pyplot as plt

def plot_grad_cam(model, layer_name, images):
    """
        Plot grad cam for the given layer for given images
    """

    for img in images:
        class_idx = np.argmax(model.predict(img[np.newaxis,:])) class_out = model[:, class_idx]
        conv_layer = model.get_layer(layer_name)
        grads = K.gradients(class_out, conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0,1,2))
        iterate = K.function([model.input], [pooled_grads, conv_layer.output[0]])

        pooled_grads, conv_layer_output = iterate([img[np.newaxis,:]])
        for i in range(pooled_grads.shape[0]):
            conv_layer_output[:,:,i] *= pooled_grads[i]

        heatmap = np.mean(conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) # Filter out negative values
        heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
        fig, ax = plt.subplots(1,2)

        ax[0].imshow(((img+0.5)*255).astype('int'))
        ax[0].set_title(f"Predicted class: {class_idx}")

        ax[1].set_title(f"GRAD CAM")
        ax[1].imshow(((img+0.5)*255).astype('int'),alpha=0.85)
        ax[1].imshow(heatmap,alpha=0.6)


