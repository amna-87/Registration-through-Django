from django.db import models
import tensorflow as tf  # Or your preferred deep learning library

def load_model():
    # Replace with your model loading code (weights, architecture)
    model = tf.keras.models.load_model('models/saved-model.h5')
    return model
