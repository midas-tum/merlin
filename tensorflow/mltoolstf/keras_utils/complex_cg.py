import functools
import tensorflow as tf
from .complex import complex_scale, complex_dot

def tf_custom_gradient_method(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, '_tf_custom_gradient_wrappers'):
            self._tf_custom_gradient_wrappers = {}
        if f not in self._tf_custom_gradient_wrappers:
            self._tf_custom_gradient_wrappers[f] = tf.custom_gradient(lambda *a, **kw: f(self, *a, **kw))
        return self._tf_custom_gradient_wrappers[f](*args, **kwargs)
    return wrapped

def cg(M, rhs):
    """
    Modified implementation of https://github.com/hkaggarwal/modl/blob/master/model.py
    M: system matrix - a function
    """
    # this is mainly for the gradient check...
    if tf.keras.backend.floatx() == 'float64':
        max_iter = 50
        tol = 1e-12
    else:
        max_iter = 10
        tol = 1e-10

    cond = lambda i, rTr, *_: tf.logical_and( tf.less(i, max_iter), rTr>tol)
    def body(i, rTr, x, r, p):
        with tf.name_scope('cgBody'):
            Ap = M(p)
            alpha = rTr / tf.math.real(complex_dot(p, Ap))
            x = x + complex_scale(p, alpha)
            r = r - complex_scale(Ap, alpha)
            rTrNew = tf.math.real(complex_dot(r, r))
            beta = rTrNew / rTr
            p = r + complex_scale(p, beta)
        return i+1, rTrNew, x, r, p

    x = tf.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = tf.math.real(complex_dot(r, r))
    loopVar = i, rTr, x, r, p
    out = tf.while_loop(cond,
                        body,
                        loopVar,
                        name='CGwhile',
                        parallel_iterations=1)[2]
    return out

class CGClass(tf.keras.layers.Layer):
    def __init__(self, A, AH):
        super().__init__()
        self.A = A
        self.AH = AH

    @tf_custom_gradient_method
    def call(self, lambdaa, x, y, *constants, training=None):
        rhs = self.AH(y, *constants) + complex_scale(x, lambdaa)

        def fn(inputs):
            rhs = inputs[0]
            constants = inputs[1:]

            def M(p):
                return self.AH(self.A(p, *constants), *constants) + complex_scale(p, lambdaa)

            out = cg(M, rhs)
            return out

        out = tf.map_fn(fn, (rhs, *constants),dtype=rhs.dtype,name='mapFn')

        def grad(e):
            #lambdaa = variables[0]
            def fn_grad(inputs):
                e = inputs[0]
                constants = inputs[1:]
                def M(p):
                    return self.AH(self.A(p, *constants), *constants) + complex_scale(p, lambdaa)
                out = cg(M, e)
                return out

            Qe = tf.map_fn(fn_grad, (e, *constants), dtype=rhs.dtype, name='mapFnGrad')
            QQe = tf.map_fn(fn_grad, (Qe, *constants), dtype=rhs.dtype, name='mapFnGrad2')
            
            dx = complex_scale(Qe, lambdaa)
            dlambdaa = tf.reduce_sum(complex_dot(Qe, x, axis=tf.range(1,tf.rank(x)))) - \
                       tf.reduce_sum(complex_dot(QQe, rhs, axis=tf.range(1,tf.rank(x))))
            dlambdaa = tf.math.real(dlambdaa)
            return [dlambdaa, dx, None] + [None for _ in constants]

        return out, grad
