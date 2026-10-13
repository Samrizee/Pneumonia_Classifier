"""Task 4 - the CNN, built from scratch (no pretrained weights), plus the bonus transfer model."""

from tensorflow.keras import layers, models

from . import config


def conv_block(filters):
    """Conv2D -> BatchNorm -> ReLU -> MaxPool."""
    return [
        layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(momentum=0.9),  # faster-adapting stats suit small datasets
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
    ]


def build_cnn(img_size=config.IMG_SIZE):
    """3 conv blocks (32 -> 64 -> 128), Flatten, Dropout(0.5), Dense(128), sigmoid output."""
    model = models.Sequential(
        [layers.Input(shape=(img_size, img_size, 1))]
        + conv_block(32)
        + conv_block(64)
        + conv_block(128)
        + [
            layers.Flatten(),
            layers.Dropout(0.5),            # before the dense layers: the data is small
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="chest_xray_cnn",
    )
    compile_model(model)
    return model


def compile_model(model):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_mobilenet(img_size=config.IMG_SIZE):
    """Bonus 2: frozen ImageNet MobileNetV2 base + a small head.

    MobileNetV2 wants 3 channels scaled to [-1, 1], so the grayscale [0, 1]
    input is converted inside the model (the rest of the pipeline is unchanged).
    Weights are downloaded from the internet the first time this is built.
    """
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 1))
    x = layers.Concatenate()([inputs, inputs, inputs])     # grayscale -> 3 channels
    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)        # [0, 1] -> [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")
    compile_model(model)
    return model
