import numpy as np
import tensorflow as tf

a=tf.constant(0.1)
x=tf.Variable(12.0)
for i in range(50):
  with tf.GradientTape() as g:
    y=x*x
  grad=g.gradient(y,x)
  x.assign(x-a*grad)
  print(x.numpy())


