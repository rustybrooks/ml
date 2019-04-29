#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os.path
import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 25
BUFFER_SIZE = 60000
BATCH_SIZE = 256

checkpoint_dir = os.path.splitext(os.path.split(os.path.realpath(__file__))[-1])[0]
checkpoint_prefix = os.path.join('.', checkpoint_dir, "ckpt")

digit_tensor = tf.convert_to_tensor([float(x) for x in range(10)])

def random_choice(a, axis, samples_shape=None):
    """

    :param a: tf.Tensor
    :param axis: int axis to sample along
    :param samples_shape: (optional) shape of samples to produce. if not provided, will sample once.
    :returns: tf.Tensor of shape a.shape[:axis] + samples_shape + a.shape[axis + 1:]
    :rtype:

    Examples:
    >>> a = tf.placeholder(shape=(10, 20, 30), dtype=tf.float32)
    >>> random_choice(a, axis=0)
    <tf.Tensor 'GatherV2:0' shape=(1, 20, 30) dtype=float32>
    >>> random_choice(a, axis=1)
    <tf.Tensor 'GatherV2_1:0' shape=(10, 1, 30) dtype=float32>
    >>> random_choice(a, axis=1, samples_shape=(2, 3))
    <tf.Tensor 'GatherV2_2:0' shape=(10, 2, 3, 30) dtype=float32
    >>> random_choice(a, axis=0, samples_shape=(100,))
    <tf.Tensor 'GatherV2_3:0' shape=(100, 20, 30) dtype=float32>
    """

    if samples_shape is None:
        samples_shape = (1,)
    shape = tuple(a.get_shape().as_list())
    dim = shape[axis]
    choice_indices = tf.random.uniform(samples_shape, minval=0, maxval=dim, dtype=tf.int32)
    samples = tf.gather(a, choice_indices, axis=axis)
    print samples
    return samples


def generate_noise(batch_size):
    noise = tf.concat(
        random_choice(digit_tensor, 0, [batch_size, 1]),
        tf.random.normal([batch_size, noise_dim])
    )
    return noise


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = generate_noise(BATCH_SIZE)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(5,5))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(int(step)))
    plt.close(fig)


def train(dataset, epochs):
    for epoch in range(epochs - step):
        print("Start epoch {}".format(int(step)))
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch + 1, seed)

        step.assign_add(1)
        if int(step) % 1 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(step), save_path))
            # print("loss {:1.2f}".format(loss.numpy()))

        print ('Time for epoch {} is {} sec'.format(int(step), time.time()-start))

    # Generate after the final epoch
    # display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
#
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
#
#
# # Batch and shuffle the data
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#
# generator = make_generator_model()
# discriminator = make_discriminator_model()
#
# # This method returns a helper function to compute cross entropy loss
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

seed = tf.Variable(generate_noise(1))
print seed[0]
# step = tf.Variable(1)
#
# checkpoint = tf.train.Checkpoint(
#     step=step,
#     generator_optimizer=generator_optimizer,
#     discriminator_optimizer=discriminator_optimizer,
#     generator=generator,
#     discriminator=discriminator,
#     seed=seed,
# )
# manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
#
# checkpoint.restore(manager.latest_checkpoint)
# if manager.latest_checkpoint:
#     print("Restored from {}".format(manager.latest_checkpoint))
# else:
#     print("Initializing from scratch.")
#
# train(train_dataset, EPOCHS)
#

