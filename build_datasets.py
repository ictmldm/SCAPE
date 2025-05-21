
import pickle 
import os
import pandas as pd
from tqdm import tqdm
import logging
from typing import Optional
import yaml
# from datetime import datetime # Uncomment if needed for log filename timestamp

# Configure logging
log_filename = f"./logs/data_preprocessing.log"
# _{datetime.now().strftime('%Y%m%d_%H%M%S')} # Uncomment if you want timestamp in log filename
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load path configuration from YAML file
# Note: The path to path_config.yaml is currently absolute.
# If the script is always run from the project root, consider changing this to a relative path like "./path_config.yaml".
with open("./path_config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Load paths from the configuration
original_news_path=config["preprocess_data"]["original_news_path"]
simplify_news_path=config["preprocess_data"]["simplify_news_path"]
test_raw_path=config["preprocess_data"]["test_raw_path"]
train_raw_path=config["preprocess_data"]["train_raw_path"]
valid_raw_path=config["preprocess_data"]["valid_raw_path"]
test_file_path=config["preprocess_data"]["test_file_path"]
train_file_path=config["preprocess_data"]["train_file_path"]
valid_file_path=config["preprocess_data"]["valid_file_path"]

# Load preprocess parameters from the configuration
MAX_CLICK_LENGTH = config["preprocess"]["max_click_length"]
LIMIT = config["preprocess"]["limit"]

def remove_unnecessary_columns(original_path):
    """
    Loads original news data, selects relevant columns, and saves to a feather file.

    Args:
        original_path (str): Path to the original news TSV file.
    """
    original_news = pd.read_csv(original_path, sep="\t")
    # Select only 'News ID', 'Headline', and 'News body' columns
    simplify_news = original_news[["News ID", "Headline", "News body"]]
    # Save the simplified data to a feather file for faster loading
    simplify_news.to_feather(simplify_news_path)
    print("Remove unnecessary columns done.")
    logging.info("Removed unnecessary columns and saved simplified news data.")

def load_feather_data(file_path) -> Optional[pd.DataFrame]:
    """
    Loads data from a feather file and fills NaN values with empty strings.

    Args:
        file_path (str): Path to the feather file.

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame, or None if file not found (not handled here, but implied by return type).
    """
    data = pd.read_feather(file_path)
    # Fill any NaN values with empty strings to avoid errors during text processing
    data.fillna(value=" ", inplace=True)
    return data

def load_tsv_data(file_path):
    """
    Loads data from a TSV file and fills NaN values with empty strings.

    Args:
        file_path (str): Path to the TSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    data = pd.read_csv(file_path, sep="\t")
    # Fill any NaN values with empty strings
    data.fillna(value=" ", inplace=True)
    return data

def build_news_dict(news_ids):
    """
    Builds a dictionary mapping news IDs to integer indices.

    Args:
        news_ids (List[str]): List of unique news IDs.

    Returns:
        Dict[str, int]: Dictionary mapping news ID to index.
                        Indices start from 1, 0 is reserved for padding.
                        'PAD' and 'UNK' tokens are included.
    """
    # Build news index dictionary starting from 1, 0 reserved for padding
    news_dict = {news_id: idx for idx, news_id in enumerate(news_ids, start=1)}
    news_dict['PAD'] = 0
    news_dict['UNK'] = len(news_dict) # Index for unknown news IDs
    return news_dict

def news_id_to_titles(click_ids, news_titles, news_dict):
    """
    Converts a list of news IDs to a list of corresponding news titles.

    Args:
        click_ids (List[str]): List of news IDs from click history.
        news_titles (Dict[int, str]): Dictionary mapping news index to title.
        news_dict (Dict[str, int]): Dictionary mapping news ID to index.

    Returns:
        List[str]: List of news titles corresponding to the input IDs.
                   Uses empty string for unknown IDs.
    """
    # Map news IDs to their corresponding titles using the news_dict and news_titles map
    titles_list = [news_titles.get(news_dict.get(id, news_dict['UNK']), "") for id in click_ids]
    return titles_list

def process_click_history(click_history_ids, news_titles, news_dict, max_click_length=50, data_type="train"):
    """
    Processes raw click history IDs, converts them to titles, and truncates to max length.

    Args:
        click_history_ids (str): Raw string of click history IDs (space or comma separated).
        news_titles (Dict[int, str]): Dictionary mapping news index to title.
        news_dict (Dict[str, int]): Dictionary mapping news ID to index.
        max_click_length (int): Maximum length of click history to retain.
        data_type (str): Type of dataset ('train' or 'test') to determine separator.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing:
            - List of click history titles (truncated).
            - List of mapped click history indices (truncated).
    """
    # Split history IDs based on data type (comma for test, space for train)
    click_history_ids = click_history_ids.split(",") if data_type == "test" else click_history_ids.split(" ")
    # Convert IDs to titles
    click_history = news_id_to_titles(click_history_ids, news_titles, news_dict)

    # Truncate click history titles to max length
    if len(click_history) > max_click_length:
        click_history = click_history[-max_click_length:]

    # Also return the corresponding list of history indices mapped from IDs
    click_history_ids_mapped = [news_dict.get(id, news_dict['UNK']) for id in click_history_ids]
    # Truncate history indices to max length
    if len(click_history_ids_mapped) > max_click_length:
        click_history_ids_mapped = click_history_ids_mapped[-max_click_length:]

    return click_history, click_history_ids_mapped


def build_test_raw(test: pd.DataFrame, news_titles, news_bodys, news_dict, max_click_length=50) -> Optional[pd.DataFrame]:
    """
    Builds the raw test dataset by processing user click history and positive news samples.

    Args:
        test (pd.DataFrame): DataFrame containing raw test data.
        news_titles (Dict[int, str]): Dictionary mapping news index to title.
        news_bodys (Dict[int, str]): Dictionary mapping news index to body.
        news_dict (Dict[str, int]): Dictionary mapping news ID to index.
        max_click_length (int): Maximum length of click history.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing the processed test samples.
    """
    print("Start building test_raw dataset")
    logging.info("Start building test_raw dataset")
    history_inputs, history_ids_list, bodys, o_titles, p_titles, test_ids = [], [], [], [], [], []
    user_ids = []
    news_count, user_count = {}, {}

    # Iterate through test data rows (userid, clicknewsID, posnewID, rewrite_titles)
    for user_id, click_history_ids, pos_ids, rewrite_titles in tqdm(test.itertuples(index=False), total=len(test)):
        # Process click history to get titles and mapped IDs
        click_history, click_history_ids_mapped = process_click_history(click_history_ids, news_titles, news_dict, max_click_length, data_type="test")
        # Split positive news IDs and corresponding rewritten titles
        pos_ids, rewrite_titles = pos_ids.split(","), rewrite_titles.split(";;")
        assert len(pos_ids) == len(rewrite_titles) # Ensure consistency

        # Process each positive news sample for the user
        for pos_id, rewrite_title in zip(pos_ids, rewrite_titles):
            # Get news body and original title using mapped index
            body = news_bodys.get(news_dict.get(pos_id, news_dict['UNK']), "")
            o_title = news_titles.get(news_dict.get(pos_id, news_dict['UNK']), "")

            # Skip samples with empty rewritten title or body
            if not rewrite_title.strip() or not body.strip():
                continue

            # Update news and user counts
            news_count[pos_id] = news_count.get(pos_id, 0) + 1
            user_count[user_id] = user_count.get(user_id, 0) + 1

            # Append data to lists
            test_ids.append(pos_id)
            user_ids.append(user_id)  # Add user_id
            history_inputs.append(click_history)
            history_ids_list.append(click_history_ids_mapped) # Add history_ids list
            bodys.append(body)
            o_titles.append(o_title)
            p_titles.append(rewrite_title)

    # Create DataFrame from collected data
    test_raw_samples = pd.DataFrame({
        'history': history_inputs,
        'history_ids': history_ids_list,  # Add history_ids column
        'user_id': user_ids,  # Add user_id column
        'news_id': test_ids,
        'bodys': bodys,
        'o_titles': o_titles,
        'p_titles': p_titles
    })
    # Save the raw test dataset
    test_raw_samples.to_feather(test_raw_path)

    # Calculate and print/log statistics
    news_count_total = len(news_count)
    user_count_total = len(user_count)
    sample_count = len(test_raw_samples)
    samples_per_user = list(user_count.values())
    average_samples_per_user = sum(samples_per_user) / len(samples_per_user) if samples_per_user else 0
    max_samples_per_user = max(samples_per_user) if samples_per_user else 0

    print(f"Test dataset stats:")
    print(f"Number of news: {news_count_total}")
    print(f"Number of users: {user_count_total}")
    print(f"Number of samples: {sample_count}")
    print(f"Average samples per user: {average_samples_per_user:.2f}")
    print(f"Max samples per user: {max_samples_per_user}")

    logging.info("Test dataset stats:")
    logging.info(f"Number of news: {news_count_total}")
    logging.info(f"Number of users: {user_count_total}")
    logging.info(f"Number of samples: {sample_count}")
    logging.info(f"Average samples per user: {average_samples_per_user:.2f}")
    logging.info(f"Max samples per user: {max_samples_per_user}")

    print(test_raw_samples.head())

    return test_raw_samples


def build_pretrain_raw(news: pd.DataFrame, train: Optional[pd.DataFrame], test: Optional[pd.DataFrame], exclude=True, save_path=None) -> Optional[pd.DataFrame]:
    """
    Builds the raw pretraining dataset from news data, optionally excluding news present in train/test sets.

    Args:
        news (pd.DataFrame): DataFrame containing simplified news data.
        train (Optional[pd.DataFrame]): Processed train DataFrame (used for exclusion).
        test (Optional[pd.DataFrame]): Processed test DataFrame (used for exclusion).
        exclude (bool): Whether to exclude news present in train/test sets.
        save_path (Optional[str]): Specific path to save the resulting feather file.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing the processed pretraining samples.
    """
    print(f"Start building pretrain_raw dataset")
    # Rename columns for consistency
    pretrain_dataset = news.rename(columns={"News ID": "news_id", "Headline": "o_titles", "News body": "bodys"})

    # Filter out news with empty or whitespace-only titles or bodies
    pretrain_dataset = pretrain_dataset.replace({"o_titles": r"^\s*$", "bodys": r"^\s*$"}, None, regex=True).dropna(subset=["o_titles", "bodys"])

    if exclude:
        logging.info("Start building pretrain_raw dataset with excluded test & train news")

        # Convert news IDs from train and test sets to sets for faster lookup
        train_news_ids = set(train["news_id"].values) if train is not None else set()
        test_news_ids = set(test["news_id"].values) if test is not None else set()

        # Exclude news present in either test or train sets
        pretrain_dataset = pretrain_dataset[~pretrain_dataset["news_id"].isin(test_news_ids)]
        pretrain_dataset = pretrain_dataset[~pretrain_dataset["news_id"].isin(train_news_ids)]

        default_filename = "pretrain_raw_ex.feather"
    else:
        logging.info("Start building pretrain_raw dataset with all news")
        default_filename = "pretrain_raw.feather"

    # Determine save path, prioritizing the provided path
    path = save_path if save_path else f"./data/processed/{default_filename}"

    # Save the pretraining dataset
    pretrain_dataset.to_feather(path)
    print(pretrain_dataset.head())

    return pretrain_dataset

def build_dataset_raw(data: pd.DataFrame, news_titles, news_bodys, news_dict, max_click_length=50, limit=None, for_rec=False, dataset_type="train") -> Optional[pd.DataFrame]:
    """
    Builds raw datasets (train, valid) by processing user click history and positive/negative news samples.

    Args:
        data (pd.DataFrame): DataFrame containing raw dataset (train or valid).
        news_titles (Dict[int, str]): Dictionary mapping news index to title.
        news_bodys (Dict[int, str]): Dictionary mapping news index to body.
        news_dict (Dict[str, int]): Dictionary mapping news ID to index.
        max_click_length (int): Maximum length of click history.
        limit (Optional[int]): Maximum number of samples per news item.
        for_rec (bool): Whether to return negative samples as titles (for recommendation) or IDs.
        dataset_type (str): Type of dataset ('train' or 'valid').

    Returns:
        Optional[pd.DataFrame]: DataFrame containing the processed samples.
    """
    print(f"Start building {dataset_type}_raw dataset")
    logging.info(f"Start building {dataset_type}_raw dataset")
    history_inputs, history_ids_list, bodys, o_titles, ids = [], [], [], [], []
    user_ids_list = []
    neg_titles_ids, neg_titles_list = [], []
    news_count, user_count = {}, {}

    # Iterate through dataset rows
    # Columns: user_id, click_history_ids, _, _, pos_ids, neg_ids, _, _, _
    for user_id, click_history_ids, _, _, pos_ids, neg_ids, _, _, _ in tqdm(data.itertuples(index=False), total=len(data)):
        # Process click history
        click_history, click_history_ids_mapped = process_click_history(click_history_ids, news_titles, news_dict, max_click_length, data_type=dataset_type)
        # Split positive and negative news IDs
        pos_ids, neg_ids = pos_ids.split(" "), neg_ids.split(" ")
        # Convert negative IDs to titles
        neg_titles = news_id_to_titles(neg_ids, news_titles, news_dict)

        # Process each positive news sample for the user
        for pos_id in pos_ids:
            # Get news body and original title using mapped index
            body = news_bodys.get(news_dict.get(pos_id, news_dict['UNK']), "")
            o_title = news_titles.get(news_dict.get(pos_id, news_dict['UNK']), "")

            # Skip samples with empty title or body
            if not o_title.strip() or not body.strip():
                continue

            # If limit is set, check if the limit for this news ID is reached
            if limit and news_count.get(pos_id, 0) >= limit:
                continue

            # Update news count (initialize to 0 if news ID not seen before)
            news_count[pos_id] = news_count.get(pos_id, 0) + 1

            # Update user count
            user_count[user_id] = user_count.get(user_id, 0) + 1

            # Append data to lists
            ids.append(pos_id)
            history_inputs.append(click_history)
            history_ids_list.append(click_history_ids_mapped) # Add history_ids column
            user_ids_list.append(user_id)  # Add user_id column
            bodys.append(body)
            o_titles.append(o_title)
            neg_titles_ids.append(neg_ids)
            neg_titles_list.append(neg_titles)

    # Depending on 'for_rec' parameter, return negative samples as titles or IDs
    neg = neg_titles_list if for_rec else neg_titles_ids

    # Build DataFrame
    dataset_raw_samples = pd.DataFrame({
        'history': history_inputs,
        'history_ids': history_ids_list,  # Add history_ids column
        'user_id': user_ids_list,  # Add user_id column
        'news_id': ids,
        'bodys': bodys,
        'o_titles': o_titles,
        'neg_titles': neg # This column name is kept for consistency, but contains either titles or IDs
    })

    # Determine save path based on dataset type, for_rec flag, and limit
    base_name = f"./data/processed/{dataset_type}_raw"
    base_name = base_name + "_for_rec" if for_rec else base_name
    dataset_raw_path = base_name + f"_limit_{limit}.feather" if limit else base_name + "_all.feather"
    # Save the dataset
    dataset_raw_samples.to_feather(dataset_raw_path)

    # Calculate and print/log statistics
    news_count_total = len(news_count)
    user_count_total = len(user_count)
    sample_count = len(dataset_raw_samples)
    samples_per_user = list(user_count.values())
    average_samples_per_user = sum(samples_per_user) / len(samples_per_user) if samples_per_user else 0
    max_samples_per_user = max(samples_per_user) if samples_per_user else 0

    print(f"{dataset_type} dataset stats:")
    print(f"Number of news: {news_count_total}")
    print(f"Number of users: {user_count_total}")
    print(f"Number of samples: {sample_count}")
    print(f"Average samples per user: {average_samples_per_user:.2f}")
    print(f"Max samples per user: {max_samples_per_user}")

    logging.info(f"{dataset_type} dataset stats:")
    logging.info(f"Number of news: {news_count_total}")
    logging.info(f"Number of users: {user_count_total}")
    logging.info(f"Number of samples: {sample_count}")
    logging.info(f"Average samples per user: {average_samples_per_user:.2f}")
    logging.info(f"Max samples per user: {max_samples_per_user}")

    print(dataset_raw_samples.head())

    return dataset_raw_samples


def main():
    """
    Main function to orchestrate the dataset building process.
    """
    # Load simplified news data
    news = load_feather_data(simplify_news_path)

    # Build news ID to index mapping
    news_ids = news["News ID"].unique().tolist()
    news_dict = build_news_dict(news_ids)

    # Build index to News ID reverse mapping
    idx_to_news_id = {idx: news_id for news_id, idx in news_dict.items()}

    # Store news titles and bodies as dictionaries for fast lookup using index
    # Ensure keys align with indices from news_dict (starting from 1)
    news_titles = {idx: title for idx, title in enumerate(news["Headline"].values, start=1)}
    news_bodys = {idx: body for idx, body in enumerate(news["News body"].values, start=1)}

    # Load test set and build test_raw
    test = load_tsv_data(test_file_path)
    build_test = build_test_raw(test, news_titles, news_bodys, news_dict, max_click_length=MAX_CLICK_LENGTH)

    # Load validation set and build valid_raw and valid_raw_for_rec
    valid = load_tsv_data(valid_file_path)
    build_valid = build_dataset_raw(valid, news_titles, news_bodys, news_dict, max_click_length=MAX_CLICK_LENGTH, limit=LIMIT, dataset_type="valid")
    build_valid_for_rec = build_dataset_raw(valid, news_titles, news_bodys, news_dict, max_click_length=MAX_CLICK_LENGTH, limit=LIMIT, dataset_type="valid", for_rec=True)

    # Load training set and build train_raw and train_raw_for_rec
    train = load_tsv_data(train_file_path)
    build_train = build_dataset_raw(train, news_titles, news_bodys, news_dict, max_click_length=MAX_CLICK_LENGTH, limit=LIMIT, dataset_type="train")
    build_train_for_rec = build_dataset_raw(train, news_titles, news_bodys, news_dict, max_click_length=MAX_CLICK_LENGTH, limit=LIMIT, for_rec=True, dataset_type="train")

    # Build pretraining datasets (with and without excluding train/test news)
    build_pretain = build_pretrain_raw(news, train=build_train, test=build_test, exclude=False)
    build_pretain_ex = build_pretrain_raw(news, train=build_train, test=None, exclude=True) # Note: exclude=True here means excluding train, but not test based on the call

    # Print/log pretrain dataset statistics
    print("Pretrain dataset stats:")
    pretrain_news_count = build_pretain['news_id'].nunique()
    pretrain_sample_count = len(build_pretain)
    print(f"Number of news: {pretrain_news_count}")
    print(f"Number of samples: {pretrain_sample_count}")

    logging.info("Pretrain dataset stats:")
    logging.info(f"Number of news: {pretrain_news_count}")
    logging.info(f"Number of samples: {pretrain_sample_count}")

    # Save mapping dictionaries to pickle files
    mappings_dir = "./data/mappings/"
    os.makedirs(mappings_dir, exist_ok=True)  # Ensure the directory exists

    # Save news_id_to_idx mapping
    news_id_to_idx_path = os.path.join(mappings_dir, 'news_id_to_idx.pkl')
    with open(news_id_to_idx_path, 'wb') as f:
        pickle.dump(news_dict, f)
    print(f"Saved News ID to index mapping to {news_id_to_idx_path}")
    logging.info(f"Saved News ID to index mapping to {news_id_to_idx_path}")

    # Save idx_to_news_id reverse mapping
    idx_to_news_id_path = os.path.join(mappings_dir, 'idx_to_news_id.pkl')
    with open(idx_to_news_id_path, 'wb') as f:
        pickle.dump(idx_to_news_id, f)
    print(f"Saved index to News ID reverse mapping to {idx_to_news_id_path}")
    logging.info(f"Saved index to News ID reverse mapping to {idx_to_news_id_path}")


if __name__ == '__main__':
    # Check if simplified news file exists, if not, create it
    if not os.path.exists(simplify_news_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(simplify_news_path), exist_ok=True)
        print(f"Directory for {simplify_news_path} created if it didn't exist.")
        logging.info(f"Directory for {simplify_news_path} created if it didn't exist.")
        remove_unnecessary_columns(original_news_path)

    # Run the main dataset building process
    main()