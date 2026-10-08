"""A small, causal two-stack Transformer for next-note prediction.

Both stacks read the same note prefix. Unlike translation, the encoder must
also be causal: an unmasked encoder would leak future labels through cross
attention even if decoder self-attention were masked.
"""

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Embedding, LayerNormalization, MultiHeadAttention


def sinusoidal_position_encoding(num_positions, d_model):
    if num_positions < 1 or d_model < 1:
        raise ValueError("Position count and model width must be positive")
    angles = np.arange(num_positions)[:, None] / np.power(
        10000, 2 * (np.arange(d_model)[None, :] // 2) / d_model
    )
    values = np.empty_like(angles)
    values[:, 0::2] = np.sin(angles[:, 0::2])
    values[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(values[None], dtype=tf.float32)


def causal_padding_mask(keys, query_length):
    """Keras masks use True for allowed (not blocked) attention positions."""
    causal = tf.range(tf.shape(keys)[1])[None, :] <= tf.range(query_length)[:, None]
    return causal[None] & tf.not_equal(keys[:, None, :], 0)


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate, cross=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.cross_attention = (
            MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads) if cross else None
        )
        self.ffn = tf.keras.Sequential([Dense(d_feedforward, activation="relu"), Dense(d_model)])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6) if cross else None
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)
        self.drop3 = Dropout(dropout_rate) if cross else None

    def call(self, x, attention_mask, context=None, cross_mask=None, training=False):
        attended = self.attention(x, x, attention_mask=attention_mask, training=training)
        x = self.norm1(x + self.drop1(attended, training=training))
        if self.cross_attention is not None:
            attended = self.cross_attention(x, context, attention_mask=cross_mask, training=training)
            x = self.norm3(x + self.drop3(attended, training=training))
        return self.norm2(x + self.drop2(self.ffn(x), training=training))


class Transformer(tf.keras.Model):
    """Predict x[t+1] using only x[:t+1]; token 0 is padding."""

    def __init__(self, num_layers, d_model, num_heads, d_feedforward,
                 input_vocab_size, target_vocab_size,
                 max_num_positions_in_pe_encoder, max_num_positions_in_pe_decoder,
                 dropout_rate=0.1):
        super().__init__()
        if min(num_layers, d_model, num_heads, d_feedforward) < 1 or d_model % num_heads:
            raise ValueError("Positive dimensions required; d_model must be divisible by num_heads")
        if min(input_vocab_size, target_vocab_size) < 2 or not 0 <= dropout_rate < 1:
            raise ValueError("Invalid vocabulary or dropout")
        self.model_config = dict(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads,
            d_feedforward=d_feedforward, input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            max_num_positions_in_pe_encoder=max_num_positions_in_pe_encoder,
            max_num_positions_in_pe_decoder=max_num_positions_in_pe_decoder,
            dropout_rate=dropout_rate,
        )
        self.enc_embedding = Embedding(input_vocab_size, d_model)
        self.dec_embedding = Embedding(target_vocab_size, d_model)
        self.enc_pe = sinusoidal_position_encoding(max_num_positions_in_pe_encoder, d_model)
        self.dec_pe = sinusoidal_position_encoding(max_num_positions_in_pe_decoder, d_model)
        self.enc_blocks = [AttentionBlock(d_model, num_heads, d_feedforward, dropout_rate)
                           for _ in range(num_layers)]
        self.dec_blocks = [AttentionBlock(d_model, num_heads, d_feedforward, dropout_rate, cross=True)
                           for _ in range(num_layers)]
        self.enc_dropout, self.dec_dropout = Dropout(dropout_rate), Dropout(dropout_rate)
        self.final_layer = Dense(target_vocab_size)
        self.scale = tf.math.sqrt(tf.cast(d_model, tf.float32))

    def call(self, input, target=None, training=False, enc_padding_mask=None,
             look_ahead_mask=None, dec_padding_mask=None):
        # Optional masks may restrict attention further, never remove causality.
        target = input if target is None else target
        tf.debugging.assert_equal(tf.shape(input), tf.shape(target),
                                  message="Both stacks require aligned note prefixes")
        length = tf.shape(input)[1]
        tf.debugging.assert_positive(length)
        tf.debugging.assert_less_equal(length, tf.shape(self.enc_pe)[1])
        tf.debugging.assert_less_equal(length, tf.shape(self.dec_pe)[1])
        enc_mask = causal_padding_mask(input, length)
        dec_mask = causal_padding_mask(target, length)
        cross_mask = causal_padding_mask(input, length)
        if enc_padding_mask is not None:
            enc_mask &= tf.cast(enc_padding_mask, tf.bool)
        if look_ahead_mask is not None:
            dec_mask &= tf.cast(look_ahead_mask, tf.bool)
        if dec_padding_mask is not None:
            cross_mask &= tf.cast(dec_padding_mask, tf.bool)
        encoded = self.enc_dropout(self.enc_embedding(input) * self.scale + self.enc_pe[:, :length],
                                   training=training)
        for block in self.enc_blocks:
            encoded = block(encoded, attention_mask=enc_mask, training=training)
        decoded = self.dec_dropout(self.dec_embedding(target) * self.scale + self.dec_pe[:, :length],
                                   training=training)
        for block in self.dec_blocks:
            decoded = block(decoded, attention_mask=dec_mask, context=encoded,
                            cross_mask=cross_mask, training=training)
        return self.final_layer(decoded)
