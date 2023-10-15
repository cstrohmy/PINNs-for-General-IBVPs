from domain import InteriorPiece, BoundaryPiece, Domain
from ibvp import IBVP
from algorithm import LearningAlgorithm
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
from numpy import cos, sin, pi


pieces = {
    'interior': InteriorPiece(
        variables=['x', 'y'],
        inequalities=[
            'x ** 2 + y ** 2 - 1'
        ],
        bounding_box=[(-1, 1), (-1, 1)],
    ),
    'boundary': BoundaryPiece(
        variables=['x', 'y'],
        parameters=['theta'],
        parameter_inequalities=[
            '-theta',
            'theta - 2 * pi'
        ],
        bounding_box=[(0, 2 * pi)],
        parametrization=[
            'cos(theta)',
            'sin(theta)'
        ],
    )
}

domain = Domain(pieces)

pdes = {
    'interior': {
        'laplacian': 'u_xx + u_yy'
    },
    'boundary': {
        'boundary': 'u - f'
    }
}

fill_ins = {'f': lambda sample: sample[:, 0] ** 2 + sample[:, 1]}


ibvp = IBVP(domain=domain, pdes=pdes, fill_ins=fill_ins, dependent_variables=['u'])


inputs = Input(shape=(2,))
latent = Dense(64, activation='relu')(inputs)
latent = Dense(64, activation='relu')(latent)
latent = Dense(64, activation='relu')(latent)
latent = Dense(64, activation='tanh')(latent)
latent = Dense(64, activation='tanh')(latent)
latent = Dense(64, activation='tanh')(latent)
outputs = Dense(1)(latent)
model = Model(inputs, outputs)


algorithm = LearningAlgorithm(ibvp)
algorithm.train(model)













