import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from ibvp import IBVP
import numpy as np
from domain import InteriorPiece, BoundaryPiece


"""
It is here that the model is actually trained. 

"""


class LearningAlgorithm:
    def __init__(self, ibvp: IBVP, loss_weights=None, sample_sizes=None, sampling_distributions=None,
                 epochs=100, learning_rate=1e-2, optimizer=Adam()):

        self.ibvp = ibvp

        if loss_weights is None:
            num_losses = len(self.piece_names)
            weights = (1 / num_losses) * np.ones(shape=(num_losses,))
            weights = tf.constant(weights, dtype=tf.float32)
            self.loss_weights = dict(zip(self.piece_names, weights))
        else:
            self.loss_weights = loss_weights

        if sample_sizes is None:
            self.sample_sizes = {}
            for piece_name, piece in self.ibvp.domain.pieces.items():
                if isinstance(piece, InteriorPiece):
                    self.sample_sizes[piece_name] = 1000
                if isinstance(piece, BoundaryPiece):
                    self.sample_sizes[piece_name] = 200
        else:
            self.sample_sizes = sample_sizes

        if sampling_distributions is None:
            self.sampling_distributions = {}
            for piece_name, piece in self.ibvp.domain.pieces.items():
                if isinstance(piece, InteriorPiece):
                    self.sampling_distributions[piece_name] = 'uniform'
                if isinstance(piece, BoundaryPiece):
                    self.sampling_distributions[piece_name] = 'uniform from parameters'
        else:
            self.sampling_distributions = sampling_distributions

        self.partial_losses = {}
        for piece_name in self.piece_names:
            self.partial_losses[piece_name] = tf.Variable(0., dtype=tf.float32)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer.learning_rate.assign(self.learning_rate)

    def train(self, model: Model):

        for epoch in range(self.epochs):

            with tf.GradientTape(persistent=True) as tape:

                # Initialize total loss
                loss = tf.Variable(0., dtype=tf.float32)

                # Loop thru domain pieces
                for piece_name in self.piece_names:

                    sample_size = self.sample_sizes[piece_name]
                    sampling_distribution = self.sampling_distributions[piece_name]
                    loss_weight = self.loss_weights[piece_name]

                    sample = self.ibvp.sample(piece_name, sample_size, sampling_distribution)
                    partial_loss = tf.Variable(0., dtype=tf.float32)
                    for pde_name in self.ibvp.pdes[piece_name].keys():
                        partial_loss = partial_loss + tf.reduce_mean(tf.square(self.ibvp.evaluate(piece_name, pde_name, sample, model, tape)))    # evaluate might be the one responsible for handling periodic boundary conditions
                    self.partial_losses[piece_name] = partial_loss / tf.constant(len(self.ibvp.pdes[piece_name]), dtype=tf.float32)

                    loss = loss + loss_weight * partial_loss

            # Compute loss gradients
            grads = tape.gradient(loss, model.trainable_variables)

            # Apply gradients
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Display learning info
            print(f'Epoch: {epoch + 1}/{self.epochs}, Loss: {loss}')

    @property
    def piece_names(self):
        return self.ibvp.piece_names







