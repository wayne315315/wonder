from const import CARDS

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class TotalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.lookup_keys = tf.constant([3, 4, 5, 6, 7], dtype=tf.int32)
        self.lookup_values = tf.constant([0, 4, 9, 15, 22], dtype=tf.int32)
        self.lookup_default = tf.constant(-100, dtype=tf.int32)

    def build(self, input_shape):
        """This method creates the sub-layers (and their weights)."""
        self.turn_emb = tf.keras.layers.Embedding(20, self.d_model, name="turn_emb")
        self.pos_emb = tf.keras.layers.Embedding(30, self.d_model, name="pos_emb")
        self.civ_emb = tf.keras.layers.Embedding(8, self.d_model, name="civ_emb")
        self.face_emb = tf.keras.layers.Embedding(3, self.d_model, name="face_emb")
        self.action_emb = tf.keras.layers.Embedding(4, self.d_model, name="action_emb")
        self.coin_emb = tf.keras.layers.Dense(self.d_model, name="coin_emb")
        super().build(input_shape)

    def call(self, inputs):
        x, card_emb_out = inputs 
        n = (tf.keras.ops.shape(x)[-2] - 6) // 19
        n = tf.convert_to_tensor(n, dtype=tf.int32)
        # Vectorized lookup
        mask = tf.keras.ops.equal(n, self.lookup_keys)
        index = tf.keras.ops.argmax(tf.cast(mask, dtype=tf.int32))
        found = tf.keras.ops.any(mask)
        value = tf.gather(self.lookup_values, index)
        o = tf.keras.ops.where(found, value, self.lookup_default)
        # Apply embeddings
        turn_emb_out = self.turn_emb(x[:, :, 0])
        # card_emb_out is the tensor passed from the shared layer
        action_emb_out = self.action_emb(x[:, :, 2])
        pos_emb_out = self.pos_emb(x[:, :, 3] + o)
        civ_emb_out = self.civ_emb(x[:, :, 4])
        face_emb_out = self.face_emb(x[:, :, 5])
        coin_emb_out = self.coin_emb(x[:, :, 6:])
        ###
        turn_emb_out = tf.cast(turn_emb_out, tf.float32)
        card_emb_out = tf.cast(card_emb_out, tf.float32)
        action_emb_out = tf.cast(action_emb_out, tf.float32)
        pos_emb_out = tf.cast(pos_emb_out, tf.float32)
        civ_emb_out = tf.cast(civ_emb_out, tf.float32)
        face_emb_out = tf.cast(face_emb_out, tf.float32)
        coin_emb_out = tf.cast(coin_emb_out, tf.float32)
        ###
        total_emb = (turn_emb_out + card_emb_out + action_emb_out + 
                     pos_emb_out + civ_emb_out + face_emb_out + coin_emb_out)
        return total_emb

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


@tf.keras.utils.register_keras_serializable()
class HandsToMask(tf.keras.layers.Layer):
    def __init__(self, num_card, **kwargs):
        super().__init__(**kwargs)
        self.num_card = num_card

    def call(self, hands):
        batch = tf.keras.ops.shape(hands)[0] 
        hands = hands - 1
        hands = hands * 3
        hands = tf.keras.ops.concatenate([hands + i for i in range(3)], axis=1)
        indices = self.h2i(hands)
        #indices = tf.map_fn(fn=lambda i: tf.map_fn(fn=lambda card: tf.stack([i, card]), elems=hands[i]), elems=tf.range(batch))
        indices = tf.reshape(indices, [-1,2])
        indices = tf.boolean_mask(indices, indices[:, 1] >= 0)
        indices, _ = tf.raw_ops.UniqueV2(x=indices, axis=tf.constant([0]))
        updates = tf.ones_like(indices[:,0], dtype=tf.float32)
        shape = tf.stack([batch, self.num_card * 3])
        scatter = tf.scatter_nd(indices, updates, shape) - 1.0
        mask = scatter * 1e2
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({"num_card": self.num_card})
        return config
    
    @staticmethod
    def h2i(hands):
        # Get the dynamic shape of the hands tensor
        shape = tf.shape(hands)
        batch_size = shape[0]
        num_cards = shape[1]
        batch_indices, _ = tf.meshgrid(
            tf.range(batch_size),
            tf.range(num_cards),
            indexing='ij'
        )
        batch_indices = tf.cast(batch_indices, dtype=hands.dtype)
        indices = tf.stack([batch_indices, hands], axis=2)
        return indices


@tf.keras.utils.register_keras_serializable()
class PredictMove(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, policy):
        # Logic from predict_move
        noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(policy), dtype=policy.dtype)))
        logits = policy + noise
        moves = tf.argsort(logits, axis=-1, direction='ASCENDING')
        return moves

    def get_config(self):
        return super().get_config()


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
    card_emb_output = hand_emb(x[:, :, 1])
    x = TotalEmbedding(d_model=d_model, name='total_emb')([x, card_emb_output])
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
    x = tf.keras.ops.sum(x, axis=1)
    features = tf.keras.layers.Dense(d_final)(x) # (batch_size, d_final)
    return features, context


# functional API model to recreate ActorCritic structure without using any subclassing
def create_ac(num_card=len(CARDS), d_final=128, d_model=256, d_ff=128, num_heads=2, num_layers=2, dropout_rate=0.0):
    # create input layers
    states = tf.keras.Input(shape=(None, 7), dtype=tf.int32)
    hands = tf.keras.Input(shape=(None,), dtype=tf.int32)
    bias = tf.constant([[-1e1 if i % 3 == 2 else 0.0 for i in range(num_card * 3)]], dtype=tf.float32)
    # compute output layers
    features, context = get_transformer(states, hands, num_card, num_layers, d_model, num_heads, d_ff, d_final, dropout_rate=dropout_rate)
    mask = HandsToMask(num_card, name='hand2mask')(hands)
    policy = get_ff(features, d_final, d_ff, dropout_rate=dropout_rate)
    policy = tf.keras.layers.Dense(num_card * 3)(policy)
    policy += tf.keras.ops.cast(bias, policy.dtype) +  tf.keras.ops.cast(mask, policy.dtype)
    value = get_ff(features, d_final, d_ff, dropout_rate=dropout_rate)
    value = tf.keras.layers.Dense(1)(value)
    moves = PredictMove(name='predict_move')(policy)
    # auxiliary outputs - scores
    scores = context
    for _ in range(num_layers):
        scores = get_encoder_layer(scores, d_model, num_heads, d_ff, dropout_rate=dropout_rate)
    scores = tf.keras.layers.GlobalMaxPooling1D()(scores)
    scores = tf.keras.layers.Dense(9)(scores)
    # recast to float32, int32 for compatibility
    policy = tf.keras.ops.cast(policy, tf.float32)
    moves = tf.keras.ops.cast(moves, tf.int32)
    value = tf.keras.ops.cast(value, tf.float32)

    inputs = [states, hands]
    outputs = [policy, moves, value, scores]
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="actor_critic")

    return model


if __name__ == "__main__":
    model = create_ac()
    model.summary()
    state = tf.ones([4,63,7], dtype=tf.int32)
    hand = tf.constant([[1,1,3,4,0,0,0] for _ in range(4)], dtype=tf.int32)
    inputs = [state, hand]
    o1s = model(inputs, training=False)
    o2s = model(inputs, training=False)
    o3s = model(inputs, training=True)
    for o1, o2, o3 in zip(o1s, o2s, o3s):
        print(o1[0, :3], o2[0, :3], o3[0, :3])
