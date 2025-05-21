import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import yaml

with open("./path_config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Replace with your model path
MODEL = "EXAMPLE_LLM"

import spacy
nlp = spacy.load("en_core_web_sm")

def count_words(text):
    """
    Counts the number of non-punctuation words in a given text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of words excluding punctuation.
    """
    if text is None or pd.isna(text):
        return 0
    doc = nlp(text)
    return len([token.text for token in doc if not token.is_punct])

def initialize_llm(model_path, temperature=0.8, top_p=0.9, max_seq_len=2048):
    """
    Initializes the LLM model and sampling parameters.

    Args:
        model_path (str): Path to the language model.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.
        max_seq_len (int): Maximum sequence length for the model.

    Returns:
        Tuple[LLM, SamplingParams]: Initialized LLM model and sampling parameters.
    """
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_seq_len_to_capture=max_seq_len
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p
    )
    return llm, sampling_params

def generate_prompts(headlines, analysis_type):
    """
    Generates prompts based on the analysis type for a list of headlines.

    Args:
        headlines (List[str]): List of unique news headlines.
        analysis_type (str): Type of analysis, "content" or "style".

    Returns:
        List[str]: List of generated prompts.

    Raises:
        ValueError: If an invalid analysis type is provided.
    """
    prompts = []
    for headline in headlines:
        if analysis_type == 'content':
            prompt = (
                f"Please analyze the potential engaging content features of the following News Headline: {headline}\n\n"
            )
        elif analysis_type == 'style':
            prompt = (
                f"Please analyze the stylistic features of the following News Headline: {headline}\n\n"
            )
        else:
            raise ValueError("Analysis type must be 'content' or 'style'")
        prompts.append(prompt)
    return prompts

def process_batches(llm, sampling_params, prompts, analysis_type, batch_size=10):
    """
    Processes prompts in batches, generates analysis results, and includes error handling.

    Args:
        llm (LLM): Initialized LLM model.
        sampling_params (SamplingParams): Sampling parameters.
        prompts (List[str]): List of prompts.
        analysis_type (str): Type of analysis, "content" or "style".
        batch_size (int): Number of prompts per batch.

    Returns:
        List[str]: List of generated analysis results. Returns None for failed prompts.
    """
    results = []
    total = len(prompts)
    for i in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i+batch_size]
        # Set system message based on analysis type
        if analysis_type == 'content':
            system_message = (
                "Role: News Content Analyst : Analyze the informative and engaging content features of a given English news headline, focusing on brevity and avoiding repetition.\n"
                "Goals: Provide a concise analysis that highlights the key content elements, including what makes the headline appealing or attention-grabbing, without repeating the headline.\n"
                "Constraints: The analysis should be brief, not exceeding 50 words, and must highlight elements that capture interest and attention.\n"
                "Skills: Proficiency in English language, ability to analyze text for engaging elements, and expertise in concise content analysis.\n"
                "Output Format: A single, concise paragraph summarizing the content features, limited to 50 words.\n"
                "Workflow: "
                "1. Carefully read and understand the given news headline. "
                "2. Identify the main content elements, such as key information, relevance, and aspects that make it engaging or appealing. "
                "3. Formulate a concise analysis that starts directly with insights, avoiding direct repetition of the headline. "
                "4. Present the analysis in a single, well-structured text paragraph.\n"
            )
        elif analysis_type == 'style':
            system_message = (
                "Role: Text Style Analyst : Analyze the stylistic features of a given English news headline, focusing on brevity and avoiding repetition.\n"
                "Goals: Identify the key stylistic elements of the news headline, ensuring the analysis is concise and does not simply restate the title.\n"
                "Constraints: The analysis should be brief, not exceeding 50 words, and must not repeat the headline verbatim.\n"
                "Skills: Proficiency in English language and literature, keen understanding of stylistic elements, and ability to write concise, precise analyses.\n"
                "Output Format: A single, concise paragraph highlighting the stylistic features, limited to 50 words.\n"
                "Workflow: "
                "1. Carefully read and understand the given news headline. "
                "2. Identify the main stylistic elements, such as tone, word choice, sentence structure, and any notable literary devices. "
                "3. Formulate a concise analysis that begins directly with observations about the style, avoiding direct repetition of the headline. "
                "4. Present the analysis in a single, well-structured paragraph.\n"
            )
        else:
            system_message = "You are a helpful assistant."

        conversations = [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ] for prompt in batch_prompts
        ]
        try:
            outputs = llm.chat(
                messages=conversations,
                sampling_params=sampling_params,
                use_tqdm=False
            )
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                results.append(generated_text)
        except Exception as e:
            # Log error and append None for each prompt in the failed batch
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            for _ in batch_prompts:
                results.append(None)
    return results

def main():
    """
    Main function to initialize LLM, load data, generate prompts, process batches,
    map results back to DataFrame, calculate word counts, and save the enhanced data.
    """
    # Initialize LLM model and sampling parameters
    llm, sampling_params = initialize_llm(MODEL)

    # Load data from the specified feather file
    simplify_news_path = config["preprocess_data"]["simplify_news_path"]
    df = pd.read_feather(simplify_news_path)

    # Ensure required columns exist in the DataFrame
    required_columns = {'News ID', 'Headline', 'News body'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Get unique headlines to avoid redundant LLM calls
    unique_headlines = df['Headline'].drop_duplicates().tolist()
    print(f"Number of unique headlines: {len(unique_headlines)}")

    # Generate prompts for content analysis
    print("Generating prompts for content analysis...")
    content_prompts = generate_prompts(unique_headlines, 'content')

    # Generate prompts for style analysis
    print("Generating prompts for style analysis...")
    style_prompts = generate_prompts(unique_headlines, 'style')

    # Set batch size for processing, adjust based on hardware capabilities
    BATCH_SIZE = 10

    # Generate content analysis results in batches
    print("Starting content analysis generation...")
    content_analyses = process_batches(llm, sampling_params, content_prompts, 'content', batch_size=BATCH_SIZE)

    # Generate style analysis results in batches
    print("Starting style analysis generation...")
    style_analyses = process_batches(llm, sampling_params, style_prompts, 'style', batch_size=BATCH_SIZE)

    # Create mappings from unique headlines to their analysis results
    headline_to_content = dict(zip(unique_headlines, content_analyses))
    headline_to_style = dict(zip(unique_headlines, style_analyses))

    # Map the analysis results back to the original DataFrame based on headlines
    df['Interest'] = df['Headline'].map(headline_to_content)
    df['Style'] = df['Headline'].map(headline_to_style)

    # Calculate word counts for Headline, Interest, and Style columns
    df['Headline Word Count'] = df['Headline'].apply(count_words)
    df['Interest Word Count'] = df['Interest'].apply(count_words)
    df['Style Word Count'] = df['Style'].apply(count_words)

    # Generate descriptive statistics for word count columns
    word_count_stats = df[['Headline Word Count', 'Interest Word Count', 'Style Word Count']].describe()
    print("\nWord Count Statistics:")
    print(word_count_stats)

    # Sort results by 'News ID' in ascending order and reset index
    results = df.sort_values('News ID').reset_index(drop=True)

    # Display the first few rows of the results DataFrame
    print("\nFirst few rows of the results DataFrame:")
    print(results.head())

    # Save the final results DataFrame to a feather file
    results.to_feather(f"./data/enhanced.feather")
    print("\nEnhanced data saved to ./data/enhanced.feather")

# Run the main program when the script is executed
if __name__ == "__main__":
    main()