from keras.optimizers import Optimizer
import keras.backend as K

class Adabound(Optimizer):
  def __init__(self, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
               epsilon=1e-8,decay=0, amsbound=False,**kwargs):
    super(Adabound, self).__init__(**kwargs)

    if not 0.0 <= lr:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= epsilon:
      raise ValueError("Invalid epsilon value: {}".format(epsilon))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    if not 0.0 <= final_lr:
      raise ValueError("Invalid final learning rate: {}".format(final_lr))
    if not 0.0 <= gamma < 1.0:
      raise ValueError("Invalid gamma parameter: {}".format(gamma))
  
    with K.name_scope(self.__class__.__name__):
      self.iterations = K.variable(0, dtype='int64', name='iterations')
      self.lr = K.variable(lr, name='lr')
      self.beta_1 = K.variable(betas[0], name='beta_1')
      self.beta_2 = K.variable(betas[1], name='beta_2')
      self.decay = K.variable(decay, name='decay')
      self.finall_lr =K.variable(final_lr,dtype='float32',name='finall_lr')
      self.gamma = K.variable(gamma, dtype='float32', name='gamma')
    
    if epsilon is None:
      epsilon = K.epsilon()
    self.epsilon = epsilon
    self.initial_decay = decay
    self.amsbound = amsbound
  
  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]
    
    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                K.dtype(self.decay))))
    
    t = K.cast(self.iterations, K.floatx()) + 1
    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                 (1. - K.pow(self.beta_1, t)))
    
    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    if self.amsbound:
      vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    else:
      vhats = [K.zeros(1) for _ in params]
    self.weights = [self.iterations] + ms + vs + vhats


    final_lr = self.finall_lr * lr_t / self.lr
    min_bound = final_lr * (1 - 1 / (self.gamma * t + 1))
    max_bound = final_lr * (1 + 1 / (self.gamma * t))
    beta_correction1 = 1 - self.beta_1 ** t
    beta_correction2 = 1 - self.beta_2 ** t
    step_size = lr_t * K.sqrt(beta_correction1) / beta_correction2
    
    for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
      if self.amsbound:
        vhat_t = K.maximum(vhat, v_t)
        p_t = K.sqrt(vhat_t) + self.epsilon
        self.updates.append(K.update(vhat, vhat_t))
      else:
        p_t = K.sqrt(v_t) + self.epsilon
      
      new_p  = p-m_t*K.clip(step_size/p_t,K.eval(min_bound),K.eval(max_bound))
      
      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(v, v_t))
      
      
      # Apply constraints.
      if getattr(p, 'constraint', None) is not None:
        new_p = p.constraint(new_p)
      
      self.updates.append(K.update(p, new_p))
    return self.updates
  
  def get_config(self):
    config = {'lr': float(K.get_value(self.lr)),
              'betas': (float(K.get_value(self.beta_1)),float(K.get_value(self.beta_2))),
              'finall_lr': float(K.get_value(self.finall_lr)),
              'gamma': float(K.get_value(self.gamma)),
              'decay': float(K.get_value(self.decay)),
              'epsilon': self.epsilon,
              'amsbound': self.amsbound}
    base_config = super(Adabound, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


from keras.layers import *
from keras.models import *

m_input = Input((32,))
m_out = Dense(10,)(m_input)
m_out = Dense(1)(m_out)
my_model =Model(m_input,m_out)

my_adabound = Adabound(lr=1e-3, betas=(0.9, 0.999),final_lr=0.1,gamma=1e-3,epsilon=1e-8,)
my_model.compile(optimizer=my_adabound,loss='mse')

