from abc import ABC, abstractmethod
import numpy as np
from numpy import cos, sin, pi
import tensorflow as tf


class DomainPiece(ABC):
    @abstractmethod
    def sample(self, num_samples, sampling_distribution):
        return


class InteriorPiece(DomainPiece):
    def __init__(self, variables: list, inequalities: list, bounding_box: list):
        self.variables = variables
        self.inequalities = inequalities
        self.bounding_box = bounding_box
        self.periodic = False

    def sample(self, num_samples, sampling_distribution='uniform'):
        if sampling_distribution == 'uniform':
            return self.sample_uniformly(num_samples)
        else:
            return None

    def sample_uniformly(self, num_samples):
        samples = []
        counter = 0
        while counter < num_samples:

            # Generate point uniformly at random from bounding box
            point = [np.random.uniform(a, b) for a, b in self.bounding_box]

            # Assign values to variables
            for j, variable in enumerate(self.variables):
                exec(f'{variable} = point[{j}]')

            # Check to see if point satisfies each inequality
            bools = []
            for inequality in self.inequalities:
                exec(f'boo = {inequality} < 0')
                exec('bools.append(boo)')

            # If point satisfies all of the inequalities, add it to samples
            if all(bools):
                samples.append(point)
                counter += 1

        samples = np.array(samples)
        samples = tf.constant(samples, dtype=tf.float32)
        return samples



class BoundaryPiece(DomainPiece):
    def __init__(self, variables, parameters, parameter_inequalities, bounding_box, parametrization):
        self.variables = variables
        self.parameters = parameters
        self.parameter_inequalities = parameter_inequalities
        self.bounding_box = bounding_box
        self.parametrization = parametrization

    def sample(self, num_samples, sampling_distribution='uniform from parameters'):
        if sampling_distribution == 'uniform from parameters':
            return self.sample_uniformly_from_parameters(num_samples)
        else:
            return None

    def sample_uniformly_from_parameters(self, num_samples):
        samples = []
        counter = 0
        while counter < num_samples:

            # Generate parameters uniformly at random from bounding box
            parameters = [np.random.uniform(a, b) for a, b in self.bounding_box]

            # Assign values to parameters
            for j, parameter in enumerate(self.parameters):
                exec(f'{parameter} = parameters[{j}]')

            # Check to see if point satisfies each parameter inequality
            bools = []
            for parameter_inequality in self.parameter_inequalities:
                exec(f'boo = {parameter_inequality} < 0')
                exec('bools.append(boo)')

            # If point satisfies all of the inequalities, add it to samples
            if all(bools):
                point = []
                for component in self.parametrization:
                    exec(f'value = {component}')
                    exec('point.append(value)')
                samples.append(point)
                counter += 1

        samples = np.array(samples)
        samples = tf.constant(samples, dtype=tf.float32)
        return samples



class PeriodicPiece(DomainPiece):
    pass







class Domain:
    def __init__(self, pieces: dict):
        self.pieces = pieces
        self.variables = list(self.pieces.values())[0].variables        # hmm should refactor, make it so that variables are same for all pieces, is necessary for PDE applications

    def sample(self, piece_name, num_samples, sampling_distribution=None):
        piece = self.pieces[piece_name]
        if sampling_distribution is None:
            if isinstance(piece, InteriorPiece):
                sampling_distribution = 'uniform'
            elif isinstance(piece, BoundaryPiece):
                sampling_distribution = 'uniform from boundary'
            else:
                TypeError('piece must be an InteriorPiece or BoundaryPiece')
        return self.pieces[piece_name].sample(num_samples, sampling_distribution)

    @property
    def piece_names(self):
        return list(self.pieces.keys())




circle = BoundaryPiece(variables=['x', 'y'],
                       parameters=['theta'],
                       parameter_inequalities=['-theta', 'theta - 2 * pi'],
                       bounding_box=[(0, 2 * pi)],
                       parametrization=['cos(theta)', 'sin(theta)'])






class Disc(Domain):
    pass

class Ball(Domain):
    pass

class Cylinder(Domain):
    pass

class Cone(Domain):
    pass


class Interval(Domain):
    pass

class Square(Domain):
    pass

class Cube(Domain):
    pass


class Torus1d(Domain):
    pass

class Torus2d(Domain):
    pass

class Torus3d(Domain):
    pass



class ConvexPolygon(Domain):
    pass





class CompositeDomain(Domain):
    """
    Add some functionality for building domains out of simpler domains.
    Includes a graph which shows how pieces are connected
    Maybe make use of magic methods to combine pieces, although that's a separate project
    """
    pass












