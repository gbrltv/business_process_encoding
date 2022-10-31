def train_text_model(model, cases):
    """
    Creates a text model

    Parameters
    -----------------------
    model,
        Text-based model containing the computed encodings
    cases: List,
        List of cases treated as sentences by the model
    Returns
    -----------------------
    model:
        Trained text-based model containing the computed encodings
    """
    model.build_vocab(cases)
    model.train(cases, total_examples=len(cases), epochs=10)

    return model
