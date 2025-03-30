import tensorflow as tf


class HandEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_card, d_model):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_card + 1, d_model) # padding value 0
    
    def call(self, x):
        return self.emb(x)


class StateEmbedding(tf.keras.layers.Layer):
    def __init__(self, hand_emb):
        super().__init__()
        d_model = hand_emb.emb.output_dim
        self.turn_emb = tf.keras.layers.Embedding(20, d_model) # turn 1-18; initial prime 0; padding value 19
        self.pos_emb = tf.keras.layers.Embedding(30, d_model) # 4+5+6+7+8 = 30; padding value 4, 9, 15, 22, 30
        #self.pos_emb = {n: tf.keras.layers.Embedding(n + 1, d_model) for n in range(3, 8)} # n : 3-7 ; padding value n
        self.civ_emb = tf.keras.layers.Embedding(8, d_model) # padding value 0
        self.face_emb = tf.keras.layers.Embedding(3, d_model) # padding value 0
        self.card_emb = hand_emb.emb # padding value 0
        self.action_emb = tf.keras.layers.Embedding(4, d_model) # padding value 0
        self.offset = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
            tf.constant([3,4,5,6,7]), 
            tf.constant([0,4,9,15,22])),
            default_value=-100
            )

    def call(self, x):
        n = (tf.shape(x)[-2] - 6) // 18
        o = int(self.offset.lookup(n).numpy())
        # (turn, card, action, pos, civ, face)
        turn_emb = self.turn_emb(x[:, :, 0])
        card_emb = self.card_emb(x[:, :, 1])
        action_emb = self.action_emb(x[:, :, 2])
        pos_emb = self.pos_emb(x[:, :, 3] + o)
        civ_emb = self.civ_emb(x[:, :, 4])
        face_emb = self.face_emb(x[:, :, 5])
        total_emb = turn_emb + card_emb + action_emb + pos_emb + civ_emb + face_emb # (bash, 18*n+6, d_model)
        return total_emb


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

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

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, hand_emb, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = StateEmbedding(hand_emb)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.embedding(x)

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x 


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x) 
        return x
  

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, hand_emb, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = hand_emb
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads,
                        dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.embedding(x)  
        x = self.dropout(x)
        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, num_card, d_final, dropout_rate=0.1):
        super().__init__()
        hand_emb = HandEmbedding(num_card, d_model)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            hand_emb=hand_emb,
                            dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            hand_emb=hand_emb,
                            dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(d_final, activation='relu')

    def call(self, *inputs):
        state, hand = inputs

        context = self.encoder(state)  # (batch_size, 18*n+6, d_model)

        x = self.decoder(hand, context)  # (batch_size, num_hand, d_model)

        # Final linear layer output.
        x = tf.reduce_sum(x, axis=1)  # (batch_size, d_model)
        features = self.final_layer(x) # (batch_size, d_final)

        """
        print("")
        print("state", state.shape)
        print("hand", hand.shape)
        print("context", context.shape)
        print("x", x.shape)
        print("features", features.shape)"
        """

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del features._keras_mask
        except AttributeError:
            pass

        return features

@tf.keras.utils.register_keras_serializable()
class ActorCritic(tf.keras.Model):
    def __init__(self, num_card, d_final, d_model=512, dff=128, num_heads=8, num_layers=4, dropout_rate=0.1):
        super().__init__()
        self.common = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, num_card=num_card, d_final=d_final, dropout_rate=dropout_rate)
        self.actor = tf.keras.layers.Dense(num_card * 3) ## TODO
        self.critic =tf.keras.layers.Dense(1) ## TODO

    def call(self, state, hand):
        features = self.common(state, hand)
        policy = self.actor(features) # logits
        value = self.critic(features) # expected return
        return policy, value