import numpy as np
from openai import OpenAI
from tqdm import tqdm


def compute_embeddings(
    client: OpenAI,
    model: str,
    texts: str | list[str],
    output_path: str | None = None,
    save_emb: bool = False,
) -> np.ndarray:
    """
    Embed a list of texts using a specified OpenAI embedding model.

    Args:
        client (OpenAI): OpenAI client class.
        model (str): Name of the embedding model to use.
        texts (str | list[str]): Contents to be embedded.
        Output_path (str | None): Path to save embeddings.
        save_emb (bool): To save embeddings or not.

    Returns:
        np.ndarray: List of embeddings corresponding to the input texts.
    """
    if isinstance(texts, list):
        embeddings = []
        for text in tqdm(texts, desc="Computing embeddings...", leave=False):
            response = client.embeddings.create(input=text, model=model)
            embeddings.append(response.data[0].embedding)

        embeddings = np.array(embeddings)

    else:
        response = client.embeddings.create(input=texts, model=model)
        embeddings = np.array(response.data[0].embedding)

    if save_emb:
        if output_path is None:
            raise ValueError("`output_path` must be provided when save_emb=True")

        np.save(output_path, embeddings)

    return embeddings


def generate_response(client: OpenAI, query: str, model: str) -> str:
    """
    Generate response given an input query.

    Args:
        client (OpenAI): OpenAI client class.
        query (str): Input query.
        model: Name of the model.

    Returns:
        str: The generaeted response.
    """
    response = client.responses.create(input=query, model=model)
    return response.output_text
