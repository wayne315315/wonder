import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class HandEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_card, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb = tf.keras.layers.Embedding(num_card + 1, d_model) # padding value 0
    
    def call(self, x):
        return self.emb(x)


@tf.keras.utils.register_keras_serializable()
class StateEmbedding(tf.keras.layers.Layer):
    def __init__(self, hand_emb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = hand_emb.emb.output_dim
        self.turn_emb = tf.keras.layers.Embedding(20, d_model) # turn 1-18; initial prime 0; padding value 19
        self.pos_emb = tf.keras.layers.Embedding(30, d_model) # offset value 0,4,9,15,22; padding 3,8,14,21,29
        self.civ_emb = tf.keras.layers.Embedding(8, d_model) # padding value 0
        self.face_emb = tf.keras.layers.Embedding(3, d_model) # padding value 0
        self.card_emb = hand_emb # padding value 0
        self.action_emb = tf.keras.layers.Embedding(4, d_model) # padding value 0
        self.coin_emb = tf.keras.layers.Dense(d_model) # padding value -1
        self.offset = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
            tf.constant([3,4,5,6,7], dtype=tf.int32), 
            tf.constant([0,4,9,15,22], dtype=tf.int32)),
            default_value=-100
            )

    def call(self, x):
        n = (tf.shape(x)[-2] - 6) // 19
        o = self.offset.lookup(n)
        # (turn, card, action, pos, civ, face, coin)
        turn_emb = self.turn_emb(x[:, :, 0])
        card_emb = self.card_emb(x[:, :, 1])
        action_emb = self.action_emb(x[:, :, 2])
        pos_emb = self.pos_emb(x[:, :, 3] + o)
        civ_emb = self.civ_emb(x[:, :, 4])
        face_emb = self.face_emb(x[:, :, 5])
        coin_emb = self.coin_emb(x[:, :, 6:])
        total_emb = turn_emb + card_emb + action_emb + pos_emb + civ_emb + face_emb + coin_emb # (batch, 19*n+6, d_model)
        return total_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hand_emb': tf.keras.utils.serialize_keras_object(self.card_emb),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        hand_emb = tf.keras.utils.deserialize_keras_object(config.pop("hand_emb"))
        return cls(hand_emb, **config)


@tf.keras.utils.register_keras_serializable()
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, key_dim, dropout=dropout)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


@tf.keras.utils.register_keras_serializable()
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


@tf.keras.utils.register_keras_serializable()
class GlobalSelfAttention(BaseAttention): 
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


@tf.keras.utils.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x


@tf.keras.utils.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, d_ff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


@tf.keras.utils.register_keras_serializable()    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, hand_emb, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.embedding = StateEmbedding(hand_emb)
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        d_ff=d_ff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'hand_emb': tf.keras.utils.serialize_keras_object(self.embedding.card_emb),
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        hand_emb = tf.keras.utils.deserialize_keras_object(config.pop("hand_emb"))
        return cls(hand_emb=hand_emb, **config)

@tf.keras.utils.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, d_ff)

    def call(self, x, context):
        x = self.self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x) 
        return x
  

@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, hand_emb, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.embedding = hand_emb
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads,
                        d_ff=d_ff, dropout_rate=dropout_rate) for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.embedding(x)  
        x = self.dropout(x)
        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'hand_emb': tf.keras.utils.serialize_keras_object(self.embedding),
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        hand_emb = tf.keras.utils.deserialize_keras_object(config.pop("hand_emb"))
        return cls(hand_emb=hand_emb, **config)

@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, num_card, d_final, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_card = num_card
        self.d_final = d_final
        self.dropout_rate = dropout_rate
        hand_emb = HandEmbedding(num_card, d_model)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, d_ff=d_ff,
                            hand_emb=hand_emb,
                            dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, d_ff=d_ff,
                            hand_emb=hand_emb,
                            dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(d_final)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'num_card': self.num_card,
            'd_final': self.d_final,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    def call(self, state, hand):
        context = self.encoder(state)  # (batch_size, 18*n+6, d_model)
        x = self.decoder(hand, context)  # (batch_size, num_hand, d_model)
        # Final linear layer output.
        x = tf.reduce_sum(x, axis=1)  # (batch_size, d_model)
        features = self.final_layer(x) # (batch_size, d_final)
        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del features._keras_mask
        except AttributeError:
            pass
        return features

@tf.keras.utils.register_keras_serializable()
class ActorCritic(tf.keras.Model):
    input_signature = (
        tf.TensorSpec(shape=[None, None, 7], dtype=tf.int32), # vs
        tf.TensorSpec(shape=[None, None], dtype=tf.int32) # hs
    )
    def __init__(self, num_card, d_final=128, d_model=256, d_ff=128, num_heads=2, num_layers=2, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_card = num_card
        self.d_final = d_final
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.common = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_card=num_card, d_final=d_final, dropout_rate=dropout_rate)
        self.actor = tf.keras.layers.Dense(num_card * 3)
        self.critic = tf.keras.layers.Dense(1)
        # introduce bias to discourage discarding
        self.bias = tf.constant([[-1e1 if i % 3 == 2 else 0.0 for i in range(num_card * 3)]], dtype=tf.float32)

    def build(self, input_shape):
        states = tf.ones([1,60,7], dtype=tf.int32)
        hands = tf.ones([1,7], dtype=tf.int32)
        features = self.common(states, hands)
        policy = self.actor(features) + self.bias
        value = self.critic(features) 

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'num_card': self.num_card,
            'd_final': self.d_final,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @tf.function
    def hands2mask(self, hands):
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
        shape = tf.stack([batch, self.num_card * 3])
        scatter = tf.scatter_nd(indices, updates, shape) - 1.0
        bias = scatter * 1e9
        return bias

    def call(self, states, hands):
        features = self.common(states, hands)
        policy = self.actor(features) + self.bias # include bias into logits to discourage discarding
        mask = self.hands2mask(hands)
        policy += mask # apply hand mask, silent non-hand cards by adding -1e9 to their logits
        value = self.critic(features) # expected return
        return policy, value
    
    @tf.function(input_signature=input_signature)
    def predict_move(self, states, hands):
        policy, _ = self(states, hands)
        # gumbel max trick to sample from policy distribution without replacement
        # http://amid.fish/humble-gumbel
        noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(policy))))
        logits = policy + noise
        moves = tf.argsort(logits, axis=-1, direction='ASCENDING')
        return moves
