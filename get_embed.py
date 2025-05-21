import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL = "EXAMPLE_EMBDEDDING_MODEL" # Replace with your model path

def load_model():
    """
    Loads and returns the SentenceTransformer embedding model.

    Returns:
        SentenceTransformer: The loaded model instance.
    """
    model = SentenceTransformer(MODEL, local_files_only=True)
    model.bfloat16()
    model.max_seq_length = 1024
    return model

def generate_embedding(model, headline, text, analysis_type):
    """
    Generates text embeddings based on the analysis type.

    Args:
        model: SentenceTransformer model instance.
        headline: News headline.
        text: Interest or Style text analysis.
        analysis_type: Type of analysis, "interest" or "style".

    Returns:
        numpy.ndarray: The embedding vector.

    Raises:
        ValueError: If an invalid analysis type is provided.
    """
    if analysis_type == "interest":
        prompt = (
            f"Instruct: Represent the query for extract key content aspects.\n"
            f"Query: Headline: '{headline}' | Analysis: '{text}'"
        )
    elif analysis_type == "style":
        prompt = (
            f"Instruct: Represent the query for identify stylistic features.\n"
            f"Query: Headline: '{headline}' | Analysis: '{text}'"
        )
    else:
        raise ValueError("Invalid analysis type. Must be 'interest' or 'style'.")

    return model.encode(prompt, prompt_name="query")

def generate_embeddings_for_dataframe(df, model):
    """
    Generates Interest and Style embedding vectors for each row in the DataFrame.

    Args:
        df: pandas DataFrame containing 'Headline', 'Interest', and 'Style' columns.
        model: SentenceTransformer model instance.

    Returns:
        pandas.DataFrame: DataFrame with 'News ID', 'Interest Embedding', and 'Style Embedding' columns.
    """
    print("Generating Interest and Style embedding vectors...")

    # Enable tqdm for progress display
    tqdm.pandas(desc="Processing Embeddings")

    df['Interest Embedding'] = df.progress_apply(
        lambda row: generate_embedding(model, row['Headline'], row['Interest'], "interest"), axis=1
    )

    df['Style Embedding'] = df.progress_apply(
        lambda row: generate_embedding(model, row['Headline'], row['Style'], "style"), axis=1
    )

    return df[['News ID', 'Interest Embedding', 'Style Embedding']]

def calculate_click_score(interest_embedding, style_embedding):
    """
    Calculates the click score as the dot product of two embedding vectors.

    Args:
        interest_embedding: Interest embedding vector.
        style_embedding: Style embedding vector.

    Returns:
        float: The dot product score.
    """
    # Calculate click score using dot product
    score = (interest_embedding @ style_embedding.T) * 100
    return score


def main():
    """
    Main function to load data, generate embeddings, calculate click scores, and save results.
    """
    # Load the enhanced data file
    df = pd.read_feather("./data/enhanced.feather")

    # Load the embedding model
    model = load_model()

    # Generate embeddings and save
    embedding_data = generate_embeddings_for_dataframe(df, model)
    print(embedding_data)

    # Calculate click score for the first data entry's embeddings
    first_row = embedding_data.iloc[0]
    interest_embedding = first_row['Interest Embedding']
    style_embedding = first_row['Style Embedding']

    click_score = calculate_click_score(interest_embedding, style_embedding)
    print(f"Click Score (Dot Product) for first data entry: {click_score}")

    # Save the embedding data
    embedding_data.to_feather("./data/interest_style_embeddings.feather")

    print("Embedding table saved to interest_style_embeddings.feather")

# Run the main program
if __name__ == "__main__":
    main()
