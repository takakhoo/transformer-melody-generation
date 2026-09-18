"""Reproduce a small, leakage-free next-note experiment and export MIDI."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from melodygenerator import MelodyGenerator, write_midi
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer


def _calculate_loss(real, pred):
    losses = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
    mask = tf.cast(real != 0, losses.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(losses * mask), tf.reduce_sum(mask))


def evaluate(model, x, y):
    logits = model(x, training=False)
    loss = float(_calculate_loss(y, logits))
    mask = y != 0
    accuracy = float(np.mean(np.argmax(logits.numpy(), axis=-1)[mask] == y[mask]))
    return {"cross_entropy": loss, "perplexity": float(np.exp(loss)), "accuracy": accuracy}


def bigram_baseline(x, y, test_x, test_y, vocab_size):
    counts = np.ones((vocab_size, vocab_size), dtype=float)  # Add-one smoothing.
    counts[:, 0] = 0
    for previous, following in zip(x.ravel(), y.ravel()):
        if following:
            counts[previous, following] += 1
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    mask = test_y != 0
    loss = float(-np.log(probabilities[test_x[mask], test_y[mask]]).mean())
    accuracy = float((np.argmax(probabilities[test_x[mask]], axis=-1) == test_y[mask]).mean())
    return {"cross_entropy": loss, "perplexity": float(np.exp(loss)), "accuracy": accuracy}


def load_artifact(directory):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    directory = Path(directory)
    model = Transformer(**json.loads((directory / "model_config.json").read_text()))
    model(tf.constant([[1, 1]]), training=False)
    model.load_weights(directory / "model.weights.h5")
    tokenizer = tokenizer_from_json((directory / "tokenizer.json").read_text())
    return model, tokenizer


def run(dataset_path="dataset.json", output="results", epochs=200, seed=7):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    preprocessor = MelodyPreprocessor(dataset_path, seed=seed)
    raw = preprocessor._load_dataset()
    melodies = [preprocessor._parse_melody(m) for m in raw]
    if len(melodies) < 2:
        raise ValueError("Need separate training and held-out melodies")
    # Split whole melodies, never overlapping windows. Melody 0 is fixed as
    # held-out before training: its notes all exist in the other five melodies.
    # Vocabulary is fitted on training data only.
    train_melodies, held_melodies = melodies[1:], melodies[:1]
    preprocessor.tokenizer.fit_on_texts(train_melodies)
    unknown = set(held_melodies[0]) - set(preprocessor.tokenizer.word_index)
    if unknown:
        raise ValueError(f"Held-out notes absent from training vocabulary: {sorted(unknown)}")
    x, y = preprocessor._create_sequence_pairs(preprocessor.tokenizer.texts_to_sequences(train_melodies))
    vx, vy = preprocessor._create_sequence_pairs(preprocessor.tokenizer.texts_to_sequences(held_melodies))
    vocab_size = len(preprocessor.tokenizer.word_index) + 1
    model = Transformer(1, 32, 2, 64, vocab_size, vocab_size, 100, 100, dropout_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(0.003, clipnorm=1.0)

    @tf.function
    def step(inputs, targets):
        with tf.GradientTape() as tape:
            loss = _calculate_loss(targets, model(inputs, training=True))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    history = [{"epoch": 0, "train": evaluate(model, x, y), "held_out": evaluate(model, vx, vy)}]
    for epoch in range(1, epochs + 1):
        step(x, y)
        if epoch % 10 == 0 or epoch == epochs:
            row = {"epoch": epoch, "train": evaluate(model, x, y), "held_out": evaluate(model, vx, vy)}
            history.append(row)
            print(f"epoch {epoch}: train CE={row['train']['cross_entropy']:.3f}, held-out CE={row['held_out']['cross_entropy']:.3f}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_config.json").write_text(json.dumps(model.model_config, indent=2) + "\n")
    (out / "tokenizer.json").write_text(preprocessor.tokenizer.to_json() + "\n")
    model.save_weights(out / "model.weights.h5")
    restored, tokenizer = load_artifact(out)
    round_trip_error = float(np.abs(model(vx).numpy() - restored(vx).numpy()).max())
    if round_trip_error > 1e-6:
        raise AssertionError("Reloaded checkpoint changed predictions")
    generator = MelodyGenerator(restored, tokenizer, max_length=32)
    prompt = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
    samples = {}
    for name, temperature in [("greedy", 0), ("sampled", 0.8)]:
        generated = generator.generate(prompt, temperature=temperature, top_k=5, seed=seed)
        samples[name] = generated
        write_midi(generated.split(), out / f"{name}.mid")
    result = dict(
        description="Six bundled melodies; five training, one held out. Educational, not a music-quality benchmark.",
        seed=seed, epochs=epochs, tensorflow=tf.__version__, parameters=model.count_params(),
        dataset_sha256=hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        train_indices=list(range(1, len(melodies))), held_out_indices=[0],
        train_notes=int(np.sum(y != 0)), held_out_notes=int(np.sum(vy != 0)),
        vocabulary_size_excluding_pad=vocab_size - 1, model_config=model.model_config,
        bigram_held_out=bigram_baseline(x, y, vx, vy, vocab_size),
        history=history, checkpoint_max_abs_error=round_trip_error, prompt=prompt, samples=samples,
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for key, label in [("train", "Training melodies"), ("held_out", "Held-out melody")]:
        ax.plot([r["epoch"] for r in history], [r[key]["cross_entropy"] for r in history], label=label)
    ax.axhline(result["bigram_held_out"]["cross_entropy"], linestyle="--", color="gray", label="Held-out bigram baseline")
    ax.set(xlabel="Full-batch updates", ylabel="Next-note cross-entropy (nats)", title="Generalization matters more than memorizing five melodies")
    ax.legend(frameon=False)
    fig.savefig(out / "learning-curve.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"final": history[-1], "bigram": result["bigram_held_out"], "checkpoint_error": round_trip_error}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.dataset, args.output, args.epochs, args.seed)
