import numpy as np
from openai import OpenAI
from tqdm import tqdm


def compute_embeddings(
    client: OpenAI,
    model: str,
    texts: str | list[str],
    output_path: str | None = None,
    save_emb: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Embed a list of texts using a specified OpenAI embedding model.

    Args:
        client (OpenAI): OpenAI client class.
        model (str): Name of the embedding model to use.
        texts (str | list[str]): Contents to be embedded.
        Output_path (str | None): Path to save embeddings.
        save_emb (bool): To save embeddings or not.
        batch_size (int): The number of chunks sent in one API request.

    Returns:
        np.ndarray: List of embeddings corresponding to the input texts.
    """
    if isinstance(texts, str):
        response = client.embeddings.create(input=texts, model=model)
        embeddings = np.array(response.data[0].embedding, dtype=np.float32)

    else:
        embeddings = []

        for start in tqdm(
            range(0, len(texts), batch_size), desc="Computing embeddings"
        ):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(input=batch, model=model)
            embeddings.extend([item.embedding for item in response.data])

        embeddings = np.array(embeddings, dtype=np.float32)

    if save_emb:
        if output_path is None:
            raise ValueError("`output_path` must be provided when save_emb=True.")

        np.save(output_path, embeddings)

    return embeddings


def generate_response(client: OpenAI, input_message: str, model: str) -> str:
    """
    Generate response given an input query.

    Args:
        client (OpenAI): OpenAI client class.
        input_message (str): Input message.
        model: Name of the model.

    Returns:
        str: The generaeted response.
    """
    response = client.responses.create(input=input_message, model=model)
    return response.output_text
