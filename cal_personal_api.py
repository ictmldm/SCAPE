import re
import pandas as pd
import yaml

# Load configuration from path_config.yaml using a relative path
# The path_config.yaml file is assumed to be in the same directory as this script.
with open("./path_config.yaml", 'r') as file:
    config = yaml.safe_load(file)

import asyncio
from openai import OpenAI
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Base URL and API key for the OpenAI client
base_url = "http://example:example/v1/"
api_key = "example.key"

# Initialize the OpenAI client
client = OpenAI(api_key=api_key, base_url=base_url)
# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

# Define paths for test data files
# These paths are relative to the current working directory
TEST_PATH = "./baseline/FPG-results.feather"
# TEST_PATH = "./baseline/SCAPE-results.feather"

# Define model parameters
MAX_MODEL_LENGTH = 2048
MODEL = "example model"
MAX_WORKERS = 50

# Define sensitive words and their replacements for text sanitization
sensitive_words = {
    "Hong Kong": "KH",
    "Taiwan": "WT",
    "Tibet": "BT",
    "Xinjiang": "JX",
    "Falun Gong": "LGF",
    "Mao Zedong": "M",
    "Zedong Mao": "M",
    "Xi Jinping": "X",
    "Jinping Xi": "X",
    "Hu Jintao": "H",
    "Jintao Hu": "H",
    "Jiang Zemin": "J",
    "Zemin Jiang": "J",
    "Deng Xiaoping": "D",
    "Xiaoping Deng": "D",
    "extradition bill": "EB bill",
    # Add more sensitive words and their replacements here
}

def replace_sensitive_words(text, replacements):
    """
    Replaces sensitive words in the input text with their defined replacements.

    Args:
        text (str): The input text string.
        replacements (dict): A dictionary mapping sensitive words to their replacements.

    Returns:
        str: The text string with sensitive words replaced.
    """
    for word, replacement in replacements.items():
        # Create a regex pattern for the whole word, case-insensitive
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        text = pattern.sub(replacement, text)
    return text

def refine_result(first_res, second_res):
    """
    Refines the comparison result based on two independent LLM responses.
    It extracts 'A' or 'B' from the responses and determines a final outcome ('win', 'lose', or 'tie').

    Args:
        first_res (str): The result from the first comparison (predict vs original).
        second_res (str): The result from the second comparison (original vs predict).

    Returns:
        str: The refined result ('win', 'lose', or 'tie').
             'win' means predict title is better, 'lose' means original title is better,
             'tie' means the results are inconsistent or invalid.
    """

    # Helper function to extract 'A' or 'B' from a response string
    def extract_result(res):
        # Use regex to find 'A' or 'B' as whole words, ignoring other text
        match = re.search(r'\b([AB])\b', res.strip())
        return match.group(1) if match else None

    # Extract clean results from both responses
    first_res_clean = extract_result(first_res)
    second_res_clean = extract_result(second_res)

    # If either extraction is invalid, return 'tie'
    if first_res_clean is None or second_res_clean is None:
        return 'tie'

    # Determine the final result based on the clean extractions
    # If the first comparison favored A (predict) and the second favored B (predict), it's a win for predict
    if first_res_clean == "A" and second_res_clean == "B":
        return 'win'
    # If the first comparison favored B (original) and the second favored A (original), it's a loss for predict
    elif first_res_clean == "B" and second_res_clean == "A":
        return 'lose'
    # Otherwise, the results are inconsistent or both favored the same, consider it a tie
    else:
        return 'tie'

async def generate_score(news_body: str, user_history: str, original_title:str ,predict_title: str):
    """
    Generates a comparison score between a predicted title and an original title
    based on user history and news body using an LLM.
    It performs two comparisons with swapped candidate positions (A and B) to improve robustness.

    Args:
        news_body (str): The body of the news article.
        user_history (str): The user's historical news clicks/titles.
        original_title (str): The original title of the news article.
        predict_title (str): The predicted title for the news article.

    Returns:
        Tuple[str, str]: A tuple containing the raw results from the first and second comparisons.
                         Returns ("Error", "Error") if an exception occurs.
    """

    try:
        # First comparison: Predicted title as Candidate A, Original title as Candidate B
        messages_first = [
            {"role": "system", "content": "You are a professional news editor skilled at analyzing users' prefered personalized news headlines."},
            {"role": "user", "content": f"""
                    Task: Compare the two candidate news headlines based on the user's historical news clicks.
                    Please respond with the serial number of the candidate title that best matches the user's interests or title style preference.
                    If the first title aligns better with the user's interests than the second title, respond with 'A'.
                    Else if the second title aligns better, respond with 'B'.\n
                    Response Format: A or B\n\n
                    User History:\n{user_history}\n\n
                    ## Candidate A: {predict_title}\n
                    ## Candidate B: {original_title}
                    """},
        ]

        # Replace sensitive words in the messages before sending to the LLM
        for message in messages_first:
            message['content'] = replace_sensitive_words(message['content'], sensitive_words)

        # Call the LLM for the first comparison
        response_first = client.chat.completions.create(
            model=MODEL,
            messages=messages_first,
            stream=False,
            max_tokens=MAX_MODEL_LENGTH,
            temperature=0.7,
        )

        # Process the first response
        if response_first and response_first.choices:
            # Get the response content and remove leading/trailing whitespace
            first_res = response_first.choices[0].message.content.strip()
        else:
            # Print error details if response is invalid
            print("Error in first comparison:", response_first.status_code if response_first else "No response")
            print("Messages sent:", messages_first[-1]['content'])
            first_res = "None" # Indicate failure

        # Second comparison: Original title as Candidate A, Predicted title as Candidate B
        messages_second = [
            {"role": "system", "content": "You are a professional news editor skilled at analyzing users' prefered personalized news headlines."},
            {"role": "user", "content": f"""
                    Task: Compare the two candidate news headlines based on the user's historical news clicks.
                    Please respond with the serial number of the candidate title that best matches the user's interests or title style preference.
                    If the first title aligns better with the user's interests than the second title, respond with 'A'.
                    Else if the second title aligns better, respond with 'B'.\n
                    Response Format: A or B\n\n
                    User History:\n{user_history}\n\n
                    ## Candidate A: {original_title}\n
                    ## Candidate B: {predict_title}
                    """},
        ]

        # Replace sensitive words in the messages before sending to the LLM
        for message in messages_second:
            message['content'] = replace_sensitive_words(message['content'], sensitive_words)

        # Call the LLM for the second comparison
        response_second = client.chat.completions.create(
            model=MODEL,
            messages=messages_second,
            stream=False,
            max_tokens=MAX_MODEL_LENGTH,
            temperature=0.7,
        )

        # Process the second response
        if response_second and response_second.choices:
            # Get the response content and remove leading/trailing whitespace
            second_res = response_second.choices[0].message.content.strip()
        else:
            # Print error details if response is invalid
            print("Error in second comparison:", response_second.status_code if response_second else "No response")
            print("Messages sent:", messages_second[-1]['content'])
            second_res = "None" # Indicate failure

        # Return both raw results
        return first_res, second_res

    except Exception as e:
        # Catch any exceptions during the process and print an error message
        print(f"\nError in generate_score: {e}")
        return "Error", "Error" # Indicate failure

# Tokenization method: split by space
def tokenize(text):
    """
    Tokenizes a text string by splitting on spaces and removing leading/trailing whitespace.

    Args:
        text (str): The input text string.

    Returns:
        List[str]: A list of tokens.
    """
    # Split the string by spaces, remove leading/trailing whitespace, and return the list of words
    return text.strip().split()

# Truncate text to a maximum number of words
def truncate_text(text, max_words):
    """
    Truncates a text string to a maximum number of words.

    Args:
        text (str): The input text string.
        max_words (int): The maximum number of words to keep.

    Returns:
        str: The truncated text string.
    """
    tokens = tokenize(text)  # Tokenize the text
    return ' '.join(tokens[:max_words])  # Truncate and rejoin into a string

# Method to truncate each title in the history list
def truncate_history(history_list, max_words_per_title=10):
    """
    Truncates each title in a list of history titles to a maximum number of words.

    Args:
        history_list (List[str]): A list of history titles.
        max_words_per_title (int): The maximum number of words per title.

    Returns:
        List[str]: A list of truncated history titles.
    """
    # Truncate each title in the history list to the maximum number of words
    return [truncate_text(title, max_words_per_title) for title in history_list]

# Multi-threaded processing function for each row of the dataframe
def process_row(row):
    """
    Processes a single row of the input DataFrame to generate a comparison result
    between the predicted and original titles using the LLM.

    Args:
        row (pd.Series): A single row from the input DataFrame.

    Returns:
        dict: A dictionary containing user_id, news_id, and the refined comparison result ('p_res').
    """
    user_id = row['user_id']
    news_id = row['news_id']
    history = row['history']
    body = row['bodys']
    predict_title = row['predictions']
    reference_title = row['references'] # Note: reference_title is not used in generate_score
    original_title = row['o_titles']

    # Truncate history and body to manageable lengths for the LLM
    truncated_history = truncate_history(history)
    truncated_body = truncate_text(body, 100)

    # Generate two independent scores using the asynchronous generate_score function
    # asyncio.run is used here because ThreadPoolExecutor does not directly support async functions
    first_res, second_res = asyncio.run(generate_score(
        news_body=truncated_body,
        user_history=truncated_history,
        original_title=original_title,
        predict_title=predict_title
    ))

    # Use refine_result to get a more accurate judgment based on the two scores
    p_res = refine_result(first_res, second_res)

    # Return the results for this row
    return {
        "user_id": user_id,
        "news_id": news_id,
        "p_res": p_res  # Result field indicating whether the predicted title 'wins', 'loses', or is a 'tie'
    }

def main():
    """
    Main function to load data, process each row using multiple threads,
    calculate statistics, and save the results.
    """
    # Load data from the specified feather file and drop duplicate rows based on user_id and news_id
    df = pd.read_feather(TEST_PATH).drop_duplicates(subset=['user_id', 'news_id'])
    # Expected columns: user_id, news_id, references, bodys, history, o_titles, predictions

    # Initialize a list to store results
    results = []

    # Use ThreadPoolExecutor for multi-threaded processing of DataFrame rows
    # max_workers is set based on the defined MAX_WORKERS constant
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the process_row function to each row of the DataFrame
        # tqdm is used to display a progress bar
        results = list(tqdm(
            executor.map(process_row, [row for _, row in df.iterrows()]),
            total=df.shape[0], # Total number of rows to process
            ncols=100, # Width of the progress bar
            desc="Processing rows" # Description for the progress bar
        ))

    # Convert the list of results (dictionaries) into a pandas DataFrame
    results_df = pd.DataFrame(results)

    # Output the statistical results (counts of 'win', 'lose', 'tie')
    p_res_counts = results_df['p_res'].value_counts()
    print("Statistical Results:")
    print(p_res_counts)

    # Define the path to save the results CSV file
    results_csv_path = "./evaluation_results.csv"
    # Save the results DataFrame to a CSV file without the index
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
    
    
if __name__ == "__main__":
    # Run the main function when the script is executed directly
    main()