from domain import InteriorPiece, BoundaryPiece, PeriodicPiece, Domain
from tensorflow import Tensor
from keras.models import Model
from typing import Union
import re
from collections import OrderedDict
import tensorflow as tf





def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_terms(expression):
    # Use regex to capture terms.
    # This pattern captures everything that is not an operator or parenthesis.
    pattern = re.compile(r'([a-zA-Z0-9_]+)')

    # Use OrderedDict to preserve the order and keep terms unique
    terms = OrderedDict()

    # Find all terms
    matches = pattern.findall(expression)
    for match in matches:
        # Skip if the term is numeric
        if not is_numeric(match):
            terms[match] = None

    return list(terms.keys())















class IBVP:
    def __init__(self, domain: Domain, pdes: dict, fill_ins: dict, dependent_variables: list):
        self.domain = domain
        self.pdes = pdes
        self.fill_ins = fill_ins
        self.dependent_variables = dependent_variables


    def evaluate(self, piece_name: str, pde_name: str, sample: Union[Tensor, tuple], model: Model, tape:tf.GradientTape): # independent variables and dependent variables also needed, but these are class attributes

        # Access partial differential equation to evaluate
        pde = self.pdes[piece_name][pde_name]

        # Define local namespace
        local_namespace = {'tf': tf, 'sample': sample, 'model': model, 'self': self, 'tape': tape}

        # Decompose pde into terms, evaluate terms
        terms = extract_terms(pde)
        for term in terms:

            if term in self.dependent_variables:
                dependent_variable_index = self.dependent_variables.index(term)
                exec(f'{term} = model(sample)[:, {dependent_variable_index}: {dependent_variable_index + 1}]', local_namespace)
            if term in self.fill_ins.keys():
                exec(f'{term} = self.fill_ins[\'{term}\'](sample)', local_namespace)
            if '_' in term:
                dependent_variable = term.split('_')[0]
                dependent_variable_index = self.dependent_variables.index(dependent_variable)
                pdivs = term.split('_')[1]

                lines = []

                # Create GradientTape lines
                for j, independent_variable in enumerate(pdivs):            # actually need backwards implementation
                    line = j * '\t' + f'with tf.GradientTape() as tape_{pdivs[:len(pdivs) - j]}:\n'
                    lines.append(line)
                    line = (j + 1) * '\t' + f'tape_{pdivs[:len(pdivs) - j]}.watch(sample)\n'
                    lines.append(line)

                # Instantiate dependent variable
                piece = self.domain.pieces[piece_name]
                if isinstance(piece, PeriodicPiece):
                    line = len(pdivs) * '\t' + f'left_sample, right_sample = piece.sample()\n'
                    lines.append(line)
                    line = len(pdivs) * '\t' + f'{dependent_variable} = model(left_sample)[:, {dependent_variable_index}] - model(right_sample)[:, {dependent_variable_index}]\n'
                    lines.append(line)
                else:
                    line = len(pdivs) * '\t' + f'{dependent_variable} = model(sample)[:, {dependent_variable_index}]\n'
                    lines.append(line)

                # Apply GradientTapes
                for j, independent_variable in enumerate(pdivs):

                    line = (len(pdivs) - j - 1) * '\t' + f'independent_variable_index = self.independent_variables.index(\'{independent_variable}\')\n'
                    lines.append(line)

                    if j == 0:
                        line = (len(pdivs) - j - 1) * '\t' + f'{dependent_variable}_{pdivs[:j + 1]} = tape_{pdivs[:j + 1]}.gradient({dependent_variable}, sample)[:, independent_variable_index: independent_variable_index + 1]\n'
                        lines.append(line)
                    else:
                        line = (len(pdivs) - j - 1) * '\t' + f'{dependent_variable}_{pdivs[:j + 1]} = tape_{pdivs[:j + 1]}.gradient({dependent_variable}_{pdivs[:j]}, sample)[:, independent_variable_index: independent_variable_index + 1]\n'
                        lines.append(line)

                code_block = lines_to_string(lines)
                exec(code_block, local_namespace)

        exec(f'result = {pde}', local_namespace)
        return local_namespace.get('result')


    def sample(self, piece_name, num_samples, sampling_distribution=None):
        return self.domain.sample(piece_name, num_samples, sampling_distribution)

    @property
    def piece_names(self):
        return self.domain.piece_names

    @property
    def independent_variables(self):
        return self.domain.variables





def lines_to_string(lines: list):
    string = ''
    for line in lines:
        string += line
    return string





'''





from domain import Domain, InteriorPiece, BoundaryPiece
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from numpy import sin, cos, pi









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



piece_name = 'interior'
pde_name = 'laplacian'
sample = domain.sample(piece_name, 10)

inputs = Input(shape=(2,))
latent = Dense(256, activation='tanh')(inputs)
latent = Dense(128, activation='tanh')(latent)
outputs = Dense(1)(latent)
model = Model(inputs, outputs)

val = ibvp.evaluate(piece_name, pde_name, sample, model)


'''



















