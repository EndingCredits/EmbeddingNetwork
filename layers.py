import tensorflow as tf

def mask_and_pool(embeds, mask, pool_type='MAX'):
    # Use broadcasting to multiply
    masked_embeds = tf.multiply(embeds, mask)

    if pool_type=='MAX':
        # Pool using max pooling
        embed = tf.reduce_max(masked_embeds, 1)
        
    if pool_type=='SoftMAX':
        # Pool using max pooling
        num_feat = embeds.get_shape().as_list()[-1]
        #blend_weights = tf.Variable(tf.random_normal((1, num_feat), stddev=5)+5, name='blend_feats')
        blend_weights = tf.Variable(tf.zeros([1, num_feat]), name='blend_feats')
        embed = tf.reduce_sum(masked_embeds * tf.nn.softmax(masked_embeds*blend_weights, 1), 1)

    elif pool_type=='AVR':
        # For mean pooling:
        embed = tf.reduce_sum(masked_embeds, 1) / tf.reduce_sum(mask, 1)
    
    elif pool_type=='COMP':
        # Complex pooling, converts tensor to a complex number, pool is then
        #  the product of all elements under complex product. Probably not very
        #  useful, but a good illustration of an alternative pooling type.
        
        # Convert to complex number
        comps = to_complex(masked_embeds)
        
        # Normalise
        norms = tf.sqrt(comps * tf.conj(comps) + 1e-6)
        comps = comps / norms
        comps = 1e-1*comps + 1.0
        
        # Mask for products:
        #comps += tf.complex(1-mask, 0.0)
        
        # Need to use scan as reduce_prod can't handle complexes
        comps = tf.transpose(comps, [1, 0, 2])
        comps = tf.scan(lambda x,y: x*y, comps)
        comps = tf.transpose(comps, [1, 0, 2])
        comp = comps[:,-1,:]

        # Convert back to real
        embed = to_real(comp)

    return embed
    
def to_complex(elems):
    old_shape = elems.get_shape().as_list()
    last = old_shape[-1]
    assert last % 2 == 0, "Final dimension must be divisible by 2"
    size = last // 2
    re, im = tf.split(elems, [size, size], axis=2)
    comp = tf.complex(re, im)
    return comp

def to_real(comp):
    re = tf.real(comp)
    im = tf.imag(comp)
    out = tf.concat([re, im], axis=1)
    return out
    
    
def att_pool(layer_size, embeds, mask, query=None, name=''):
    import common_attention

    # Get bias from mask
    bias = tf.transpose(tf.expand_dims((1 - mask)* -1e9, 1), [0, 1, 3, 2])
    if query is None:
        query = tf.Variable([[1.0]*128]*64)
    query = tf.expand_dims(query, axis=1)
    out = common_attention.multihead_attention(
            query, embeds, bias, 32, layer_size,
            layer_size, 8, 0.0, name=name+'attn')
    out = tf.squeeze(out, axis=1)

    return out

    
# Gets the indexes specified by the lengths, 
#  equivalent to doing output[:, lengths, :]
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

def self_attention_layer(state, mask, key_size=64, name=''):
    in_size = state.get_shape().as_list()[2]
    num_elems = state.get_shape().as_list()[1]
    
    combined, params = invariant_layer(2*key_size+in_size, state)
    q, k, v = tf.split(combined, [key_size, key_size, in_size], axis=2)
    #v = state
        
    q *= key_size**-0.5 * mask
    
    logits = tf.matmul(q, k, transpose_b=True)

    
    weights = tf.nn.softmax(logits)
    out = state + tf.matmul(weights, v)
    
    return out, params
    

def self_attention_layer_deepmind(out_size, state, mask, key_size=64, value_size=64, name=''):
    in_size = state.get_shape().as_list()[2]
    num_elems = state.get_shape().as_list()[1]
    
    combined, params_1 = invariant_layer(2*key_size + value_size, state)
    q, k, v = tf.split(combined,
        [key_size, key_size, value_size], axis=2)
    
    #combined, params_1 = invariant_layer(2*key_size, state)
    #q, k = tf.split(combined, [key_size, key_size], axis=2)
    #v = state
        
    q *= key_size**-0.5
    
    logits = tf.matmul(q, k, transpose_b=True)
    logits *= mask
    
    weights = tf.nn.softmax(logits)
    attended = tf.matmul(weights, v) + v
    
    out, params_2 = invariant_layer(out_size, attended)

    return out, params_1 + params_2
    
    
def relation_layer(out_size, state, mask, name=''):
    in_size = state.get_shape()[2]
    num_elems = tf.shape(state)[1]
    
    flat_shape = [-1, num_elems*num_elems, out_size]
    mat_shape = [-1, num_elems, num_elems, out_size]
    
    combined, params_1 = invariant_layer(2*out_size, state)
    q, k = tf.split(combined, [out_size, out_size], axis=2)
    
    qk = tf.expand_dims(q, -3) + tf.expand_dims(k, -2)
    #qk = tf.reshape(qk, flat_shape)
    #qk = tf.reshape(qk, mat_shape)
    
    mask_ = tf.expand_dims(mask, -3) * tf.expand_dims(mask, -2)
    qk_ = qk - (1-mask_)*10e9
    
    qk = tf.nn.softmax(qk_, dim=3)

    out = tf.reduce_max(qk, 2)# / tf.reduce_sum(mask_, -2)
    
    return out, params_1
