#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi, Olexa Bilaniuk
#
# Note: The implementation of complex Batchnorm is based on
#       the Keras implementation of batch Normalization
#       available here:
#       https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py
# Note2: This implementation is from https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py

import numpy as np
from   tensorflow.keras.layers import Layer
from   tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K
import tensorflow as tf

def sqrt_init(shape, dtype=None):
    value = (1 / np.sqrt(2)) * K.ones(shape)
    return value


def sanitizedInitGet(init):
    if init in ["sqrt_init"]:
        return sqrt_init
    else:
        return initializers.get(init)
def sanitizedInitSer(init):
    if init in [sqrt_init]:
        return "sqrt_init"
    else:
        return initializers.serialize(init)

class ComplexBatchNormalization(Layer):
    """Complex version of the real domain 
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous complex layer at each batch,
    i.e. applies a transformation that maintains the mean of a complex unit
    close to the null vector, the 2 by 2 covariance matrix of a complex unit close to identity
    and the 2 by 2 relation matrix, also called pseudo-covariance, close to the 
    null matrix.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=2` in `ComplexBatchNormalization`.
        momentum: Momentum for the moving statistics related to the real and
            imaginary parts.
        epsilon: Small float added to each of the variances related to the
            real and imaginary parts in order to avoid dividing by zero.
        center: If True, add offset of `beta` to complex normalized tensor.
            If False, `beta` is ignored.
            (beta is formed by real_beta and imag_beta)
        scale: If True, multiply by the `gamma` matrix.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the real_beta and the imag_beta weight.
        gamma_diag_initializer: Initializer for the diagonal elements of the gamma matrix.
            which are the variances of the real part and the imaginary part.
        gamma_off_initializer: Initializer for the off-diagonal elements of the gamma matrix.
        moving_mean_initializer: Initializer for the moving means.
        moving_variance_initializer: Initializer for the moving variances.
        moving_covariance_initializer: Initializer for the moving covariance of
            the real and imaginary parts.
        beta_regularizer: Optional regularizer for the beta weights.
        gamma_regularizer: Optional regularizer for the gamma weights.
        beta_constraint: Optional constraint for the beta weights.
        gamma_constraint: Optional constraint for the gamma weights.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self,
                 channel_last=True,
                 momentum=0.9,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer='sqrt_init',
                 gamma_off_initializer='zeros',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='sqrt_init', 
                 moving_covariance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 beta_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.channel_last = channel_last
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer              = sanitizedInitGet(beta_initializer)
        self.gamma_diag_initializer        = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer         = sanitizedInitGet(gamma_off_initializer)
        self.moving_mean_initializer       = sanitizedInitGet(moving_mean_initializer)
        self.moving_variance_initializer   = sanitizedInitGet(moving_variance_initializer)
        self.moving_covariance_initializer = sanitizedInitGet(moving_covariance_initializer)
        self.beta_regularizer              = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer        = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer         = regularizers.get(gamma_off_regularizer)
        self.beta_constraint               = constraints .get(beta_constraint)
        self.gamma_diag_constraint         = constraints .get(gamma_diag_constraint)
        self.gamma_off_constraint          = constraints .get(gamma_off_constraint)
        self.reduction_axes = [0]

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.channel_last:
            reduction_axes = list(range(1, ndim-1))
        else:
            reduction_axes = list(range(2, ndim))
        self.reduction_axes += reduction_axes
        self.reduction_axes = [np.mod(a, ndim) for a in self.reduction_axes]
        self.param_axes = [a for a in range(ndim) if a not in self.reduction_axes]
        self.param_shape = [input_shape[a] if a in self.param_axes else 1 for a in range(ndim)]

        # always scale and center!
        self._moving_mean = self.add_weight(shape=self.param_shape + [2,],
                                        initializer=self.moving_mean_initializer,
                                        name='moving_mean',
                                        trainable=False)
                                        
        self.moving_Vrr = self.add_weight(shape=self.param_shape,
                                            initializer=self.moving_variance_initializer,
                                            name='moving_Vrr',
                                            trainable=False)
        self.moving_Vii = self.add_weight(shape=self.param_shape,
                                            initializer=self.moving_variance_initializer,
                                            name='moving_Vii',
                                            trainable=False)
        self.moving_Vri = self.add_weight(shape=self.param_shape,
                                            initializer=self.moving_covariance_initializer,
                                            name='moving_Vri',
                                            trainable=False)

        if self.scale:
            self.gamma_rr = self.add_weight(shape=self.param_shape,
                                            name='gamma_rr',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ii = self.add_weight(shape=self.param_shape,
                                            name='gamma_ii',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape=self.param_shape,
                                            name='gamma_ri',
                                            initializer=self.gamma_off_initializer,
                                            regularizer=self.gamma_off_regularizer,
                                            constraint=self.gamma_off_constraint)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None

        if self.center:
            self._beta = self.add_weight(shape=self.param_shape + [2,],
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self._beta = None

        self.built = True

    @property
    def beta(self):
        return tf.complex(self._beta[...,0], self._beta[...,1])

    @property
    def moving_mean(self):
        return tf.complex(self._moving_mean[...,0], self._moving_mean[...,1])

    def call(self, inputs, training=None):
        # always normalize
        mu = K.mean(inputs, axis=self.reduction_axes)
        mu = K.reshape(mu, self.param_shape)
 
        input_train_centered = inputs - mu
        xre = tf.math.real(input_train_centered)
        xim = tf.math.imag(input_train_centered)

        Vrr = K.var(xre, axis=self.reduction_axes) + self.epsilon
        Vrr = K.reshape(Vrr, self.param_shape)

        Vii = K.var(xim, axis=self.reduction_axes) + self.epsilon
        Vii = K.reshape(Vii, self.param_shape)
        
        # Vri contains the real and imaginary covariance for each feature map.
        Vri = K.mean(xre * xim, axis=self.reduction_axes)
        Vri = K.reshape(Vri, self.param_shape)

        input_bn_train = self.whiten(input_train_centered,
                                    Vrr,
                                    Vii,
                                    Vri)
        if training:
            update_list = []
            mu_2ch = tf.concat([tf.expand_dims(tf.math.real(mu), -1),
                                tf.expand_dims(tf.math.imag(mu), -1)], -1)

            update_list.append(K.moving_average_update(self._moving_mean, mu_2ch, self.momentum))
            update_list.append(K.moving_average_update(self.moving_Vrr, Vrr, self.momentum))
            update_list.append(K.moving_average_update(self.moving_Vii, Vii, self.momentum))
            update_list.append(K.moving_average_update(self.moving_Vri, Vri, self.momentum))

        input_bn_inference = self.whiten(inputs - self.moving_mean,
                                        self.moving_Vrr,
                                        self.moving_Vii,
                                        self.moving_Vri)

        # Pick the normalization form corresponding to the training phase.
        return K.in_train_phase(input_bn_train,
                                input_bn_inference,
                                training=training)

    def whiten(self, inputs, cov_uu, cov_vv, cov_uv):
        # scale
        xre = tf.math.real(inputs)
        xim = tf.math.imag(inputs)
        cov_vu = cov_uv

        # 3. get R = [[p, q], [r, s]], with E R c c^T R^T = R M R = I
        sqrdet = K.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
        denom = sqrdet * K.sqrt(cov_uu + 2 * sqrdet + cov_vv)

        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom

        # 4. apply R to x (manually)
        out_re = xre * p + xim * r
        out_im = xre * q + xim * s

        # Scaling   | gamma_rr   gamma_ri | * | out_re |
        #           | gamma_ri   gamma_ii |   | out_im |
        # Shifting: beta (tf.complex)           
        if self.scale:
            out_bn_re = self.gamma_rr * out_re + self.gamma_ri * out_im
            out_bn_im = self.gamma_ri * out_re + self.gamma_ii * out_im
            out_bn = tf.complex(out_bn_re, out_bn_im)
        else:
            out_bn = tf.complex(out_re, out_im)

        if self.center:
            return out_bn + self.beta
        else:
            return out_bn

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer':              sanitizedInitSer(self.beta_initializer),
            'gamma_diag_initializer':        sanitizedInitSer(self.gamma_diag_initializer),
            'gamma_off_initializer':         sanitizedInitSer(self.gamma_off_initializer),
            'moving_mean_initializer':       sanitizedInitSer(self.moving_mean_initializer),
            'moving_variance_initializer':   sanitizedInitSer(self.moving_variance_initializer),
            'moving_covariance_initializer': sanitizedInitSer(self.moving_covariance_initializer),
            'beta_regularizer':              regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer':        regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer':         regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint':               constraints .serialize(self.beta_constraint),
            'gamma_diag_constraint':         constraints .serialize(self.gamma_diag_constraint),
            'gamma_off_constraint':          constraints .serialize(self.gamma_off_constraint),
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
