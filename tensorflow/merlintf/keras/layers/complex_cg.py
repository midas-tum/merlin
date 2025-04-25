import functools
import tensorflow as tf
import merlintf

def tf_custom_gradient_method(f):
    """
    Decorator. Allows to implement custom gradient for a class 
    call function. Args are fully supported. Kwargs are only supported in eager
    mode.
    """
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, '_tf_custom_gradient_wrappers'):
            self._tf_custom_gradient_wrappers = {}
        if f not in self._tf_custom_gradient_wrappers:
            self._tf_custom_gradient_wrappers[f] = tf.custom_gradient(lambda *a: f(self, *a)) # kwargs currently only supported in eager mode
        #     self._tf_custom_gradient_wrappers[f] = tf.custom_gradient(lambda *a, **kw: f(self, *a, **kw))
        return self._tf_custom_gradient_wrappers[f](*args)  # kwargs currently only supported in eager mode
        # return self._tf_custom_gradient_wrappers[f](*args, **kwargs)
    return wrapped

def cg(M, rhs, max_iter, tol):
    """
    Conjugate gradient (CG) algorithm.
    Modified version of https://github.com/hkaggarwal/modl/blob/master/model.py

    Args:
        M (function handle): system matrix
        rhs (tensor): Right-hand-side of the linear system of equations that is
            solved.
        max_iter (int): Maximal number of iterations for the CG algorithm.
        tol (float): Stopping criterion for the CG algorithm.

    Returns:
        Tensor: Result of the CG
    """
    cond = lambda i, rTr, *_: tf.logical_and( tf.less(i, max_iter), rTr > tol)
    def body(i, rTr, x, r, p):
        with tf.name_scope('cgBody'):
            Ap = M(p)
            alpha = rTr / tf.math.real(merlintf.complex_dot(p, Ap))
            x = x + merlintf.complex_scale(p, alpha)
            r = r - merlintf.complex_scale(Ap, alpha)
            rTrNew = tf.math.real(merlintf.complex_dot(r, r))
            beta = rTrNew / rTr
            p = r + merlintf.complex_scale(p, beta)
        return i + 1, rTrNew, x, r, p

    x = tf.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = tf.math.real(merlintf.complex_dot(r, r))
    loopVar = i, rTr, x, r, p
    out = tf.while_loop(cond,
                        body,
                        loopVar,
                        name='CGwhile',
                        parallel_iterations=1)[2]
    return out

class CGClass(tf.keras.layers.Layer):
    def __init__(self, A, AH, max_iter=10, tol=1e-10, parallel_iterations=None):
        """Run the conjugate gradient (CG) algorithm on a given pair of forward/
           adjoint operators A/AH

        Args:
            A (function handle): Forward operator
            AH (function handle): Adjoint operator
            max_iter (int, optional): Maximal number of iterations for the CG 
                algorithm. Defaults to 10.
            tol (float, optional): Stopping criterion for the CG algorithm. 
                Defaults to 1e-10.
            parallel_iterations (int, optional): Number of iterations that the 
                map functions run in parallel. Defaults to None.
        """
        super().__init__()
        self.A = A
        self.AH = AH
        self.max_iter = max_iter
        self.tol = tol
        self.parallel_iterations = parallel_iterations

    @tf_custom_gradient_method
    def call(self, lambdaa, x, y, *constants, training=None):
        def fn(inputs):
            x = inputs[0]
            y = inputs[1]
            constants = inputs[2:]
            rhs = self.AH(y, *constants) + merlintf.complex_scale(x, lambdaa)

            def M(p):
                return self.AH(self.A(p, *constants), *constants) + \
                       merlintf.complex_scale(p, lambdaa)

            out = cg(M, rhs, self.max_iter, self.tol)
            return out, rhs

        out, rhs = tf.map_fn(fn, (x, y, *constants), 
                            fn_output_signature=(x.dtype, x.dtype),
                            name='mapFn',
                            parallel_iterations=self.parallel_iterations)

        def grad(e):
            #lambdaa = variables[0]
            def fn_grad(inputs):
                e = inputs[0]
                constants = inputs[1:]
                def M(p):
                    return self.AH(self.A(p, *constants), *constants) + \
                           merlintf.complex_scale(p, lambdaa)
                Qe = cg(M, e, self.max_iter, self.tol)
                QQe = cg(M, Qe, self.max_iter, self.tol)
                return Qe, QQe

            Qe, QQe = tf.map_fn(fn_grad, (e, *constants),
                            fn_output_signature=(x.dtype, x.dtype), 
                            name='mapFnGrad', 
                            parallel_iterations=self.parallel_iterations)

            dx = merlintf.complex_scale(Qe, lambdaa)
            dlambdaa = tf.reduce_sum(merlintf.complex_dot(Qe, x, axis=tf.range(1,tf.rank(x)))) - \
                       tf.reduce_sum(merlintf.complex_dot(QQe, rhs, axis=tf.range(1,tf.rank(x))))
            dlambdaa = tf.math.real(dlambdaa)
            return [dlambdaa, dx, None] + [None for _ in constants]

        return out, grad
