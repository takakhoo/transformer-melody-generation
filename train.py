"""Train the repository's compact encoder-decoder melody Transformer.

The script builds a tokenized ``tf.data`` input pipeline, optimizes a masked
cross-entropy objective for ten epochs, and prints one autoregressive example.
The bundled dataset is intentionally small, so the output is a learning
artifact rather than evidence of production-quality music generation.
"""

import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer

# Advanced Training Configuration Parameters
EPOCHS = 10                    # Number of complete training cycles through the dataset
BATCH_SIZE = 32               # Optimal batch size for GPU memory efficiency
DATA_PATH = "dataset.json"    # Path to our musical training dataset
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 100  # Maximum sequence length for positional encoding

# Sophisticated Loss Function and Optimizer Configuration
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"  # Advanced loss computation for musical sequences
)
optimizer = Adam()  # Adaptive learning rate optimizer for stable musical learning


def train(train_dataset, transformer, epochs):
    """
    Trains the Transformer model on a given dataset for a specified number of epochs.

    Parameters:
        train_dataset (tf.data.Dataset): The training dataset.
        transformer (Transformer): The Transformer model instance.
        epochs (int): The number of epochs to train the model.
    """
    print("Training the model...")
    for epoch in range(epochs):
        total_loss = 0
        # Iterate over each batch in the training dataset
        for (batch, (input, target)) in enumerate(train_dataset):
            # Perform a single training step
            batch_loss = _train_step(input, target, transformer)
            total_loss += batch_loss
            print(
                f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy()}"
            )


@tf.function
def _train_step(input, target, transformer):
    """
    Performs a single training step for the Transformer model.

    Parameters:
        input (tf.Tensor): The input sequences.
        target (tf.Tensor): The target sequences.
        transformer (Transformer): The Transformer model instance.

    Returns:
        tf.Tensor: The loss value for the training step.
    """
    # Prepare the target input and real output for the decoder
    # Pad the sequences on the right by one position
    target_input = _right_pad_sequence_once(target[:, :-1])
    target_real = _right_pad_sequence_once(target[:, 1:])

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Forward pass through the transformer model
        # TODO: Add padding mask for encoder + decoder and look-ahead mask
        # for decoder
        predictions = transformer(
            input,
            target_input,
            training=True,
            enc_padding_mask=None,
            look_ahead_mask=None,
            dec_padding_mask=None,
        )

        # Compute loss between the real output and the predictions
        loss = _calculate_loss(target_real, predictions)

    # Calculate gradients with respect to the model's trainable variables
    gradients = tape.gradient(loss, transformer.trainable_variables)

    # Apply gradients to update the model's parameters
    gradient_variable_pairs = zip(gradients, transformer.trainable_variables)
    optimizer.apply_gradients(gradient_variable_pairs)

    # Return the computed loss for this training step
    return loss


def _calculate_loss(real, pred):
    """
    Computes the loss between the real and predicted sequences.

    Parameters:
        real (tf.Tensor): The actual target sequences.
        pred (tf.Tensor): The predicted sequences by the model.

    Returns:
        average_loss (tf.Tensor): The computed loss value.
    """

    # Compute loss using the Sparse Categorical Crossentropy
    loss_ = sparse_categorical_crossentropy(real, pred)

    # Create a mask to filter out zeros (padded values) in the real sequences
    boolean_mask = tf.math.equal(real, 0)
    mask = tf.math.logical_not(boolean_mask)

    # Convert mask to the same dtype as the loss for multiplication
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Apply the mask to the loss, ignoring losses on padded positions
    loss_ *= mask

    # Calculate average loss, excluding the padded positions
    total_loss = tf.reduce_sum(loss_)
    number_of_non_padded_elements = tf.reduce_sum(mask)
    average_loss = total_loss / number_of_non_padded_elements

    return average_loss


def _right_pad_sequence_once(sequence):
    """
    Pads a sequence with a single zero at the end.

    Parameters:
        sequence (tf.Tensor): The sequence to be padded.

    Returns:
        tf.Tensor: The padded sequence.
    """
    return tf.pad(sequence, [[0, 0], [0, 1]], "CONSTANT")


if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    train_dataset = melody_preprocessor.create_training_dataset()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1,
    )

    train(train_dataset, transformer_model, EPOCHS)

    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer
    )
    start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
    new_melody = melody_generator.generate(start_sequence)
    print(f"Generated melody: {new_melody}")
