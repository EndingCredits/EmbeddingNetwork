import tensorflow as tf

def mask_and_pool(embeds, mask, pool_type='max_pool'):
    # Use broadcasting to multiply
    masked_embeds = tf.multiply(embeds, mask)

    # Pool using max pooling
    embed = tf.reduce_max(masked_embeds, 1)

    # For mean pooling:
    #embed = tf.reduce_sum(masked_embeds, 1) / tf.reduce_sum(mask, 1)

    return embed
    
# Gets the indexes specified by the lengths, equivalent to doing output[:, lengths, :]
def last_relevant(output, lengths):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (lengths - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def invariant_layer(out_size, inputs, context=None, name=''):
    in_size = inputs.get_shape().as_list()[-1]
    if context is not None:
      context_size = context.get_shape().as_list()[-1]

    with tf.variable_scope(name) as vs:
      w_e = tf.Variable(tf.random_normal((in_size,out_size), stddev=0.1), name='w_e')
      if context is not None:
        w_c = tf.Variable(tf.random_normal((context_size,out_size), stddev=0.1), name='w_c')
      b = tf.Variable(tf.zeros(out_size), name='b')

    if context is not None:
       context_part = tf.expand_dims(tf.matmul(context, w_c), 1)
    else:
       context_part = 0
    
    element_part = tf.nn.conv1d(inputs, [w_e], stride=1, padding="SAME")

    elements = element_part + context_part + b

    params = [w_e, w_c, b] if context is not None else [w_e, b]

    # Returns elements and the params
    return elements, params
    
    
def fc_layer(out_size, state, name=''):
    in_size = state.get_shape().as_list()[-1]

    with tf.variable_scope(name) as vs:
      w = tf.Variable(tf.random_normal((in_size, out_size), stddev=0.1), name='w_e')
      b = tf.Variable(tf.zeros(out_size), name='b')

    out = tf.matmul(state, w) + b
    
    params = [w, b]

    return out, params
