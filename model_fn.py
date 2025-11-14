import tensorflow as tf


def get_total_emb(x, hand_emb, turn_emb, pos_emb, civ_emb, face_emb, action_emb, coin_emb, offset):
    # compute output layer
    n = (tf.keras.ops.shape(x)[-2] - 6) // 19
    n = tf.convert_to_tensor(n, dtype=tf.int32)
    o = offset.lookup(n)
    
    # (turn, card, action, pos, civ, face, coin)
    turn_emb_out = turn_emb(x[:, :, 0])
    card_emb_out = hand_emb(x[:, :, 1])
    action_emb_out = action_emb(x[:, :, 2])
    pos_emb_out = pos_emb(x[:, :, 3] + o)
    civ_emb_out = civ_emb(x[:, :, 4])
    face_emb_out = face_emb(x[:, :, 5])
    coin_emb_out = coin_emb(x[:, :, 6:])
    
    # Add all embeddings
    total_emb = (turn_emb_out + card_emb_out + action_emb_out + 
                 pos_emb_out + civ_emb_out + face_emb_out + coin_emb_out)
    return total_emb


def get_ff(x, d_model, d_ff, dropout_rate=0.1):
    # create layers
    seq = tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    add = tf.keras.layers.Add()
    layer_norm = tf.keras.layers.LayerNormalization()
    # compute output layer
    x = add([x, seq(x)])
    x = layer_norm(x)
    return x


def get_global_self_attention(x, num_heads, key_dim, dropout=0.0):
    # create layers
    mha = tf.keras.layers.MultiHeadAttention(num_heads, key_dim, dropout=dropout)
    layernorm = tf.keras.layers.LayerNormalization()
    add = tf.keras.layers.Add()
    # compute output layer
    attn_output = mha(
        query=x,
        value=x,
        key=x
    )
    x = add([x, attn_output])
    x = layernorm(x)
    return x


def get_cross_attention(x, context, num_heads, key_dim, dropout=0.0):
    # create layers
    mha = tf.keras.layers.MultiHeadAttention(num_heads, key_dim, dropout=dropout)
    layernorm = tf.keras.layers.LayerNormalization()
    add = tf.keras.layers.Add()
    # compute output layer
    attn_output = mha(
        query=x,
        key=context,
        value=context
    )
    x = add([x, attn_output])
    x = layernorm(x)
    return x


def get_encoder_layer(x, d_model, num_heads, d_ff, dropout_rate=0.1):
    x = get_global_self_attention(x, num_heads, d_model, dropout=dropout_rate)
    x = get_ff(x, d_model, d_ff, dropout_rate=dropout_rate)
    return x


def get_encoder(x, hand_emb, num_layers, d_model, num_heads, d_ff, dropout_rate=0.1):
    d_model = hand_emb.output_dim
    turn_emb = tf.keras.layers.Embedding(20, d_model, name="turn_emb")
    pos_emb = tf.keras.layers.Embedding(30, d_model, name="pos_emb")
    civ_emb = tf.keras.layers.Embedding(8, d_model, name="civ_emb")
    face_emb = tf.keras.layers.Embedding(3, d_model, name="face_emb")
    action_emb = tf.keras.layers.Embedding(4, d_model, name="action_emb")
    coin_emb = tf.keras.layers.Dense(d_model, name="coin_emb")
    offset = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
        tf.constant([3,4,5,6,7], dtype=tf.int32), 
        tf.constant([0,4,9,15,22], dtype=tf.int32)),
        default_value=-100
    )
    
    # The Lambda now calls the logic function and passes the layers
    x = tf.keras.layers.Lambda(
        lambda t: get_total_emb(
            t, hand_emb, turn_emb, pos_emb, civ_emb, 
            face_emb, action_emb, coin_emb, offset
        ), 
        name='total_emb', 
        output_shape=(None, d_model)
    )(x)
    
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    for _ in range(num_layers):
        x = get_encoder_layer(x, d_model, num_heads, d_ff, dropout_rate=dropout_rate)
    return x


def get_decoder_layer(x, context, d_model, num_heads, d_ff, dropout_rate=0.1):
    x = get_global_self_attention(x, num_heads, d_model, dropout=dropout_rate)
    x = get_cross_attention(x, context, num_heads, d_model, dropout=dropout_rate)
    x = get_ff(x, d_model, d_ff, dropout_rate=dropout_rate)
    return x


def get_decoder(x, context, hand_emb, num_layers, d_model, num_heads, d_ff, dropout_rate=0.1):
    x = hand_emb(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    for _ in range(num_layers):
        x = get_decoder_layer(x, context, d_model, num_heads, d_ff, dropout_rate=dropout_rate)
    return x


def get_transformer(state, hand, num_card, num_layers, d_model, num_heads, d_ff, d_final=128, dropout_rate=0.1):
    hand_emb = tf.keras.layers.Embedding(num_card + 1, d_model)
    context = get_encoder(state, hand_emb, num_layers, d_model, num_heads, d_ff, dropout_rate=dropout_rate)
    x = get_decoder(hand, context, hand_emb, num_layers, d_model, num_heads, d_ff, dropout_rate=dropout_rate)
    #x = tf.reduce_sum(x, axis=1)  # (batch_size, d_model)
    x = tf.keras.ops.sum(x, axis=1)
    features = tf.keras.layers.Dense(d_final)(x) # (batch_size, d_final)
    return features


@tf.function
def hands2mask(hands, num_card):
    batch = tf.shape(hands)[0]
    hands -= 1
    hands *= 3
    hands = tf.concat([hands + i for i in range(3)], axis=1)
    indices = tf.map_fn(fn=lambda i: tf.map_fn(fn=lambda card: tf.stack([i, card]), elems=hands[i]), elems=tf.range(batch))
    indices = tf.reshape(indices, [-1,2])
    # filter out invalid indices
    indices = tf.boolean_mask(indices, indices[:, 1] >= 0)
    indices, _ = tf.raw_ops.UniqueV2(x=indices, axis=tf.constant([0]))
    updates = tf.ones_like(indices[:,0], dtype=tf.float32)
    shape = tf.stack([batch, num_card * 3])
    scatter = tf.scatter_nd(indices, updates, shape) - 1.0
    mask = scatter * 1e9
    return mask


@tf.function
def predict_move(policy):
    # gumbel max trick to sample from policy distribution without replacement
    # http://amid.fish/humble-gumbel
    noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(policy))))
    logits = policy + noise
    moves = tf.argsort(logits, axis=-1, direction='ASCENDING')
    return moves


# functional API model to recreate ActorCritic structure without using any subclassing
def create_ac(num_card, d_final=128, d_model=256, d_ff=128, num_heads=2, num_layers=2, dropout_rate=0.1):
    # create input layers
    states = tf.keras.Input(shape=(None, 7), dtype=tf.int32)
    hands = tf.keras.Input(shape=(None,), dtype=tf.int32)
    bias = tf.constant([[-1e1 if i % 3 == 2 else 0.0 for i in range(num_card * 3)]], dtype=tf.float32)
    # compute output layers
    features = get_transformer(states, hands, num_card, num_layers, d_model, num_heads, d_ff, d_final, dropout_rate=dropout_rate)
    mask = tf.keras.layers.Lambda(lambda t: hands2mask(t, num_card), name='hand2mask', output_shape=(num_card * 3,))(hands)
    policy = get_ff(features, d_final, d_ff, dropout_rate=dropout_rate)
    policy = tf.keras.layers.Dense(num_card * 3)(policy) + bias + mask
    value = get_ff(features, d_final, d_ff, dropout_rate=dropout_rate)
    value = tf.keras.layers.Dense(1)(value)
    moves = tf.keras.layers.Lambda(lambda t: predict_move(t), name='predict_move', output_shape=(num_card * 3,))(policy)

    inputs = [states, hands]
    outputs = [features, policy, value, moves]
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="actor_critic")

    return model


if __name__ == "__main__":
    from const import CARDS

    num_card = len(CARDS)
    model = create_ac(num_card, dropout_rate=0.1)
    model.summary()

    state = tf.ones([4,63,7], dtype=tf.int32)
    hand = tf.ones([4,21], dtype=tf.int32)
    inputs = [state, hand]
    outputs = model(inputs, training=True)
    outputs_ = model(inputs, training=True)

    for output, output_ in zip(outputs, outputs_):
        print(output[0][:3], output_[0][:3])
