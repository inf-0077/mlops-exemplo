
from tensorflow.keras import layers, Sequential, optimizers
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
from dvc.api import params_show

class Training:
    def __init__(self, dvclive, params):
        self.dvclive = dvclive
        self.params = params
        self.model_filepath = params['model_filepath']
        self.batch_size = params['batch_size']
        self.hwc = (
            self.params['input_height'],
            self.params['input_width'],
            self.params['input_channels'],
        )
        self.hw = (
            self.params['input_height'],
            self.params['input_width'],
        )

    def build_model(self):

        self.model = Sequential([
            layers.Conv2D(6, (5, 5), activation='relu', input_shape=self.hwc),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(120, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        return self

    def load_ds(self):
        # Data Augumentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            channel_shift_range=10,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_ds = train_datagen.flow_from_directory(
            './ds/train',
            target_size=self.hw,
            interpolation='bicubic',
            class_mode='binary',
            shuffle=True,
            batch_size=self.batch_size
        )

        val_datagen = ImageDataGenerator(rescale=1.0/255.0)
        val_ds = val_datagen.flow_from_directory(
            './ds/valid',
            target_size=self.hw,
            class_mode='binary',
            shuffle=False,
            batch_size=self.batch_size
        )
        return train_ds, val_ds

    def train(self):
        train_ds, val_ds = self.load_ds()

        self.model.fit(
            train_ds,
            steps_per_epoch=train_ds.samples // self.batch_size,
            validation_data=val_ds,
            validation_steps=val_ds.samples // self.batch_size,
            epochs=self.params['epochs'],
            verbose=self.params['verbose'],
            callbacks=[DVCLiveCallback(self.dvclive)]
        )

        return self

    def save_model(self):
        pathlib.Path(self.model_filepath).unlink(missing_ok=True)

        self.model.save(self.model_filepath)

        self.dvclive.log_artifact(
            self.model_filepath,
            type='model',
            desc='model',
            name='model',
        )


def main():
    params = params_show()

    with Live() as dvclive:
        Training(dvclive, params).build_model().train().save_model()


if __name__ == '__main__':
    main()
