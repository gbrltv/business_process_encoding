def train_model(model, cases):
    """
    Creates a text model

    Parameters
    -----------------------
    model: List,
        Text-based model containing the computed encodings
    cases,
        List of cases treated as sentences by the model
    Returns
    -----------------------
    vectors: List
        list of vector encodings for each trace
    """
    model.build_vocab(cases)
    model.train(cases, total_examples=len(cases), epochs=10)

    return model
