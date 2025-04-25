import tensorflow as tf
import merlintf
from merlintf.keras.layers.complex_cg import CGClass

class DCGD(tf.keras.layers.Layer):
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, trainable=True,
                name='dc-gd', **kwargs):
        """Gradient Descent Data Consistency (DCGD) for a given pair of
           forward/adjoint operators A/AH.

        Args:
            A (function handle): Forward operator
            AH (function handle): Adjoint operator
            weight_init (float, optional): Initialization for data term weight.
                Defaults to 1.0.
            weight_scale (float, optional): Scale that is multiplied to weight. 
                Might be helpful to make training of the weight faster.
                Defaults to 1.0.
            name (str, optional): Name of the layer. Defaults to 'dc-gd'.
        Kwargs:
            parallel_iterations: Defines how many instances in a batch are processed
                in parallel. Defaults to None, i.e., whole batch is processed at once.
        """
        super().__init__()

        parallel_iterations = kwargs.get('parallel_iterations', None)
        if parallel_iterations != None:
            def A_call(x, *constants):
                def A_fn(inputs):
                    return A(*inputs)
                out = tf.map_fn(A_fn, (x, *constants),
                            name='mapForward', 
                            fn_output_signature=x.dtype,
                            parallel_iterations=parallel_iterations)
                return out

            def AH_call(x, *constants):
                def AH_fn(inputs):
                    return AH(*inputs)
                out = tf.map_fn(AH_fn, (x, *constants), 
                            name='mapAdjoint',
                            fn_output_signature=x.dtype,
                            parallel_iterations=parallel_iterations)
                return out

            self.A = A_call
            self.AH = AH_call
        else:
            self.A = A
            self.AH = AH

        self.weight_init = weight_init
        self.weight_scale = weight_scale
        self.weight_trainable = trainable
    
    def build(self, input_shape):
        self._weight = self.add_weight(name='weight',
                shape=(1,),
                constraint=tf.keras.constraints.NonNeg(),
                initializer=tf.keras.initializers.Constant(self.weight_init),
                trainable=self.weight_trainable)

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def call(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        return x - merlintf.complex_scale(self.AH(self.A(x, *constants) - y, *constants), 
                                 self.weight * scale)


class DCPM(tf.keras.layers.Layer):
    """Proximal Mapping Data Consistency (DCPM) for a given pair of
        forward/adjoint operators A/AH. Runs the conjugate gradient algorithm to
        solve the proximal mapping, see Aggarwal et al. (2018) for more details.

    Args:
        A (function handle): Forward operator
        AH (function handle): Adjoint operator
        weight_init (float, optional): Initialization for data term weight.
            Defaults to 1.0.
        weight_scale (float, optional): Scale that is multiplied to weight. 
            Might be helpful to make training of the weight faster.
            Defaults to 1.0.
        name (str, optional): Name of the layer. Defaults to 'dc-pm'.
    Kwargs:
        parallel_iterations: Defines how many instances in a batch are processed
            in parallel. Defaults to None equals Default in eager mode: 1 default in graph mode: 10
    """
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, trainable=True, name='dc-pm', 
                **kwargs):
        super().__init__()
        self.A = A
        self.AH = AH
        max_iter = kwargs.get('max_iter', 10)
        tol = kwargs.get('tol', 1e-10)
        parallel_iterations = kwargs.get('parallel_iterations', None)
        self.prox = CGClass(A, AH, max_iter=max_iter, tol=tol, parallel_iterations=parallel_iterations)

        self.weight_init = weight_init
        self.weight_scale = weight_scale
        self.weight_trainable = trainable

    def build(self, input_shape):
        self._weight = self.add_weight(name='weight',
                shape=(1,),
                constraint=tf.keras.constraints.NonNeg(),
                initializer=tf.keras.initializers.Constant(self.weight_init),
                trainable=self.weight_trainable)

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def call(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        lambdaa = 1.0 / tf.math.maximum(self.weight * scale, 1e-9)
        return self.prox(lambdaa, x, y, *constants)

class itSENSE(tf.keras.layers.Layer):
    """Iterative SENSE.

    Args:
        A (function handle): Forward operator
        AH (function handle): Adjoint operator
        weight (float, optional): Regularization weight. Defaults to 0.0.
        name (str, optional): Name of the layer. Defaults to 'itSENSE'.
    Kwargs:
        parallel_iterations: Defines how many instances in a batch are processed
            in parallel. Defaults to None equals Default in eager mode: 1 default in graph mode: 10
    """
    def __init__(self, A, AH, weight=0.0, name='itSENSE', 
                **kwargs):
        super().__init__()
        self.A = A
        self.AH = AH
        self.max_iter = kwargs.get('max_iter', 10)
        self.tol = kwargs.get('tol', 1e-10)
        parallel_iterations = kwargs.get('parallel_iterations', None)
        self.op = CGClass(A, AH, max_iter=self.max_iter, tol=self.tol, parallel_iterations=parallel_iterations)

        self.weight_init = weight

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',
                shape=(1,),
                constraint=tf.keras.constraints.NonNeg(),
                initializer=tf.keras.initializers.Constant(self.weight_init),
                trainable=False)

    def call(self, inputs):
        y = inputs[0]
        constants = inputs[1:]
        x = tf.zeros_like(self.AH(y, *constants))
        return self.op(self.weight, x, y, *constants)
