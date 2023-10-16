from domain import InteriorPiece, BoundaryPiece, PeriodicPiece, Domain
from tensorflow import Tensor
from keras.models import Model
from typing import Union
import re
from collections import OrderedDict
import tensorflow as tf


"""
The IBVP class is in some sense the "computational core" of the project. The main computation that needs to be performed
is to evaluate a given pde on a given domain piece, where the "aspiring" solution is a neural network. Once we are
able to evaluate the pde and are careful to ensure that the result is still recognized by tensorflow as a differentiable
function of the model parameters, it is easy to construct a loss function which enforces that the pde be satisfied by
the network. This task is taken up in algorithm.py
"""


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
    """
    The IBVP class contains all of the information needed to specify an initial boundary value problem.
    """
    def __init__(self, domain: Domain, pdes: dict, fill_ins: dict, dependent_variables: list):
        """
        Collects the domain, any terms needed to be "filled in", and names of dependent variables
        :param domain: Domain instance, essentially a dictionary of domain pieces
        :param pdes: dictionary where the keys are the names of domain pieces, the values are dictionaries such that
        the keys are names for the partial differential equation, the values are the actual equation
        :param fill_ins: dictionary where keys are the names of the domain pieces, values are dictionaries such that
        the keys are names of any term needed to be filled in (e.g. variable coefficients, forcing terms), and the
        values are the actual callables
        :param dependent_variables: list of strings of the dependent variables (note independent variables are
        identical to the variables attribute of self.domain
        """
        self.domain = domain
        self.pdes = pdes
        self.fill_ins = fill_ins
        self.dependent_variables = dependent_variables

    def evaluate(self, piece_name: str, pde_name: str, sample: Union[Tensor, tuple], model: Model, tape:tf.GradientTape):
        """
        The basic computation needed is to evaluate the PDE/PDE systems on all of the domain pieces, on a given sample
        :param piece_name: name of a domain piece
        :param pde_name: name of a pde defined on the domain piece, which we will be evaluating
        :param sample: tensor or tuple of data points
        :param model: neural network which will be trained to solve the pde
        :param tape: a gradient tape used to update the model parameters
        :return:
        """

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



