import tensorflow as tf
from functools import reduce
import numpy as np

################################################################################
###################### Standard network layers #################################
################################################################################


def linear(x, out_size, **kwargs):
    return tf.layers.dense(x, out_size, **kwargs)

def conv2d(*args, **kwargs):
    return tf.layers.conv2d(*args, **kwargs)

################################################################################
######################## Set network layers ####################################
################################################################################


def pool(x,
         mask=None,
         keepdims=False,
         pool_type='max'):
    """
    Applies some pooling function along the penultimate dimension, and applies a
    mask where appropriate.
    
    It is assumed that x is of shape .... x N x C where N is the number of
    elements and C is the number of channels. 
    """

    if pool_type == 'max':
        if mask is not None:
            x = x * mask - (1.0 - mask)*10e9
        out = tf.reduce_max(x, -2, keepdims)
        
    elif pool_type == 'sum':
        if mask is not None:
            x = x * mask
        out = tf.reduce_sum(x, -2, keepdims)
            
    elif pool_type == 'mean':
        if mask is not None:
            x = x * mask
            x_sum = tf.reduce_sum(x, -2, keepdims)
            out = x_sum / tf.reduce_sum(mask, -2, keepdims)
        else:
            out = tf.reduce_mean(x, -2, keepdims)

    elif pool_type == 'std':
        if mask is not None:
            x = x * mask
            x_mean = pool(x, mask=mask, pool_type="mean", keepdims=keepdims)
            x_sq_sum = tf.reduce_sum(x*x, -2, keepdims)
            var = x_sq_sum / tf.reduce_sum(mask, -2, keepdims) - x_mean * x_mean
            out = tf.sqrt(var)
        else:
            out = tf.math.reduce_std(x, -2, keepdims)

    return out


def normalize(x,
              mask=None):
    """
    Normalises by mean and std
    """

    if mask is None:
       mask = get_mask(x) # Get mask directly from state

    x = x - pool(x, mask, keepdims=True, pool_type="mean")
    x = x / (pool(x, mask, keepdims=True, pool_type="std") + 10e-6)
    out = x * mask

    return out


def equiv_submax(x,
                 mask=None,
                 name='submax'):
    """
    The 'equivariant' transformation used in the "Deep Sets" paper. N.B: In the
    paper it is combined with a linear transformation (and is actually a special
    case of the more general 'context-concatenation' equivariant layer) but we
    find it more useful to use this as a separate layer.
    
    It is assumed that x is of shape .... x N x C where N is the number of
    elements and C is the number of channels. 
    """
    
    out = pool(x, mask, keepdims=True) - x
    
    return out


def transform_layer(x,
                    context,
                    mask=None):
    """
    The 'feature transform' used in "pointnet paper. Applies a linear layer to
    the context, and then reshapes this into a C x C matrix and applies a matrix
    transformation to all elements in x.
    
    It is assumed that x is of shape .... x N x C where N is the number of
    elements and C is the number of channels, and context is of the shape ... x
    C'
    """

    # Get C x C matrix from context
    C = shape_list(x)[-1]
    transform = reshape_range( linear(context, C * C), [C, C], -1)

    # Apply transform to elements
    x = tf.matmul(x, transform)

    return x


def attn_qkv(x1,
             x2,
             key_size,
             value_size=None,
             num_heads=1,
             mask=None,
             reuse=None,
             use_mlp_attn=False,
             name='attn_qkv'):
    """
    Adapted from tensor2tensor library. MLP attention from
    https://arxiv.org/abs/1409.0473

    The set of vectors x1 'attends' to the set x2. For self attn, set x1 = x2.
    
    It is assumed that x is of shape .... x N x C where N is the number of
    elements and C is the number of channels. 
    
    TODO: add initializer, and activation functions
    """
    value_size = key_size if value_size is None else value_size
    
    with tf.variable_scope(name):

        q = linear(x1, key_size, use_bias=False, reuse=reuse, name='query')
        k = linear(x2, key_size, use_bias=False, reuse=reuse, name='key')
        v = linear(x2, value_size, use_bias=False, reuse=reuse, name='value')
      
        if num_heads != 1:
            def split_heads(x, num_heads):
                new_channels = shape_list(x)[-1] // num_heads
                x = reshape_range(x, [num_heads, new_channels])
                return transpose(x, [-3, -2])

            q = split_heads(q, num_heads)
            k = split_heads(k, num_heads)
            v = split_heads(v, num_heads)

            if mask is not None:
                mask = tf.expand_dims(mask, -3)
                #Will be the same mask for every head


        def dot_product_attn(q, k, v, mask=None):
            key_depth_per_head = key_size // num_heads
            q *= key_depth_per_head**-0.5

            logits = tf.matmul(q, k, transpose_b=True)
            # ... x l_q x l_k x heads x heads (?)
            if mask is not None:
                bias = (1-mask)*10e9
                logits = logits + bias
            weights = tf.nn.softmax(logits)

            # v: ... x l_q x l_k x heads x channels
            return tf.matmul(weights, v)


        def mlp_attn(q, k, v, mask=None):
            split_dim = -3 if num_heads != 1 else -2
            
            # Get q_i + k_j matrix
            q = reshape_range(q, [-1, 1], split_dim)
            k = reshape_range(k, [1, -1], split_dim)
            h = tf.nn.tanh(q + k)

            # Get raw logits
            logits = linear(h, 1, reuse=reuse, use_bias=False, name='mlp')
            logits = tf.squeeze(logits, -1) # ... x l_q x l_k x heads 
            
            # Get attn weights
            if mask is not None:
                bias = (1-mask)*10e9
                logits = logits + bias
            weights = tf.nn.softmax(logits)

            # v: ... x l_q x l_k x heads x channels
            return tf.matmul(weights, v, transpose_a=True)


        if use_mlp_attn:
            v_out = mlp_attn(q, k, v, mask)
        else:
            v_out = dot_product_attn(q, k, v, mask)

        def join_heads(x):
            return reshape_range(transpose(x, [-3, -2]), [value_size], -2, 2)
            
        if num_heads != 1:
            v_out = join_heads(v_out)
    
    return v_out


def kary_pooling(x,
                 k,
                 layers,
                 num_samples=None,
                 mask=None,
                 pool_type='max',
                 activation=tf.nn.relu,
                 initializer=tf.truncated_normal_initializer(0, 0.02),
                 name='k-ary_pooling'):
    """
    Inspired by k-ary Janossy pooling.
    """

    # Rename for brevity
    act_fn = activation

    with tf.variable_scope(name):


        if num_samples is None:
            # Do full k-ary pooling

            # First layer - abuse broadcasting
            def get_transform(i):
                y = linear(x, layers[0], name='layer_0.' + str(i),
                           kernel_initializer=initializer)
                return reshape_range(y, [1]*i+[-1]+[1]*(k-i-1), -2)
            x = reduce(tf.add, map(get_transform , range(k)))
            x = act_fn(x) 

            # Rest of layers
            for i, layer in enumerate(layers[1:]):
                x = linear(x, layer, name='layer_' + str(i+1),
                           activation=act_fn, kernel_initializer=initializer)

            # Pool
            for i in range(k):
                x = pool(x, reshape_range(mask, [1]*(k-i)), pool_type=pool_type)

        else:
            # Sample random permutations
            m = mask if mask is not None else tf.ones(shape_list(x)[:-1]+[1])

            # Sample random permutations
            def get_random_idx(mask, k=k, n=num_samples):
                assert len(np.shape(mask)) == 2
                selected = []
                ids = np.arange(np.shape(mask)[1])
                offset = 0
                for row in mask:
                    sel = []
                    for i in range(n):
                        p = row / np.sum(row, -1)
                        picked = np.random.choice(ids, k, False, p=p)
                        sel.append(picked + offset)
                    selected.append(np.concatenate(sel))
                    offset += len(row)
                return np.array(selected)
            selected = tf.py_func(get_random_idx,
                                  [tf.squeeze(m,-1)],
                                  tf.int64)

            old_shape = shape_list(x)
            selected = tf.reshape(selected, old_shape[:-2]+[num_samples * k])
            x = reshape_range(x, [-1], 0, to=-1)
            x = tf.gather(x, selected)
            x = reshape_range(x, [num_samples, k*old_shape[-1]], -2, 2)

            # layers
            for i, layer in enumerate(layers[1:]):
                x = linear(x, layer, name='layer_' + str(i),
                           activation=act_fn, kernel_initializer=initializer)

            # Pool
            x = pool(x, pool_type=pool_type)

    return x


################################################################################
########################### General utils ######################################
################################################################################


def shape_list(x):
    """
    Taken from tensor2tensor library.
    
    Return list of dims, statically where possible.
    """
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def reshape_range(x, new_shape, dim_start=-1, num_dims=1, to=None):
    """
    Split's the dim_start'th dimension into a new set of dimensions of shape
    new_shape.
    A range of dims can be reshaped at once by changing num_dims, or by setting
    to to the final dimension to be reshaped
    """

    in_shape = shape_list(x)
    dim_start = dim_start % len(in_shape)
    dim_end = dim_start+num_dims if to is None else (to % len(in_shape))
    out_shape = in_shape[:dim_start] + new_shape + in_shape[dim_end:]

    return tf.reshape(x, out_shape)

def transpose(x, transpose_dims):
    """
    More comprehensive transpose wrapper
    """

    num_in_dims = len(shape_list(x))
    out_dims = list(range(num_in_dims))

    for i, _ in enumerate(transpose_dims):
        j = (i+1) % len(transpose_dims)
        dim_i = transpose_dims[i] % num_in_dims
        dim_j = transpose_dims[j] % num_in_dims
        # swap ith and jth dim in out dims
        out_dims[dim_i] = dim_j

    return tf.transpose(x, out_dims)
    

def get_mask(x):
    """
    Returns a matrix with values set to 1 where elements aren't padding
    Assumes input is of the form [...] x C, and that empy inputs are all 0 hence
    we return a matrix of shape [...] x 1 with 0's in locations where last
    dimension is all 0, and 1 elsewhere. (We keep dim for broadcasting).
    """
    emb_sum = tf.reduce_sum(tf.abs(x), axis=-1, keep_dims=True)
    mask = 1.0 - tf.to_float(tf.equal(emb_sum, 0.0))
    return tf.stop_gradient(mask)

def get_lengths(x, is_mask=False):
    """
    Returns the number of non-zero elements in each batch.
    Assumes input is of the form [...] x N x C, and that empy inputs are all 0
    hence we return a matrix of shape [...] x 1 where values are the number of non-
    zero elements.
    """
    mask = x if is_mask else get_mask(x)
    lens = tf.reduce_sum(mask, axis=-2)
    return tf.stop_gradient(lens)


def last_relevant(output, lengths):
    """
    Gets the indexes specified by the lengths, 
    equivalent to doing output[:, lengths, :]
    Only works for BxLxC inputs...
    """
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (lengths - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
    
    
def combine_weights(in_list):
    """
    Returns a 1D tensor of the input list of (nested lists of) tensors, useful
    for doing things like comparing current weights with old weights for EWC.
    
    1.) For all elements in input list, (ln 3)
          if a list combine it recursively 
          else leave it alone
    2.) From resulting list, get all non-none elements and flatten them (ln 2)
    3.) If resulting list is empty return None (ln 1)
          else return concatenation of list
    ( All on one line :) )
    """
    
    return (lambda x: None if not x else tf.concat(x, axis=0)) (
        [ tf.reshape(x, [-1]) for x in
        [ combine_weights(x) if isinstance(x, list) else x for x in in_list ]
        if x is not None])



################################################################################
############################ Legacy code #######################################
################################################################################


def __mask_and_pool(embeds, mask, pool_type='MAX'):
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
    
def __to_complex(elems):
    old_shape = elems.get_shape().as_list()
    last = old_shape[-1]
    assert last % 2 == 0, "Final dimension must be divisible by 2"
    size = last // 2
    re, im = tf.split(elems, [size, size], axis=2)
    comp = tf.complex(re, im)
    return comp

def __to_real(comp):
    re = tf.real(comp)
    im = tf.imag(comp)
    out = tf.concat([re, im], axis=1)
    return out
    
    
def __att_pool(layer_size, embeds, mask, query=None, name=''):
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

def __invariant_layer(out_size, inputs, context=None, name=''):
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
    
    
def __fc_layer(out_size, state, name=''):
    in_size = state.get_shape().as_list()[-1]

    with tf.variable_scope(name) as vs:
      w = tf.Variable(tf.random_normal((in_size, out_size), stddev=0.1), name='w_e')
      b = tf.Variable(tf.zeros(out_size), name='b')

    out = tf.matmul(state, w) + b
    
    params = [w, b]

    return out, params

def __self_attention_layer(state, mask, key_size=64, name=''):
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
    

def __self_attention_layer_deepmind(out_size, state, mask, key_size=64, value_size=64, name=''):
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
    
    
def __relation_layer(out_size, state, mask, name=''):
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
