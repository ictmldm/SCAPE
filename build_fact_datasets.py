import os
from typing import Optional
import pandas as pd
import pickle
import random
import spacy
import numpy as np
from tqdm import tqdm
import torch
from multiprocessing import Pool
import logging
import yaml
from build_datasets import build_news_dict, load_feather_data, load_tsv_data, process_click_history
from matrics import fact_score

# Load configuration file
# Changed absolute path to relative path
with open("./path_config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    
LLM_PATH = "EXAMPLE_PATH" # Replace with your actual path

# Configure paths
simplify_news_path = config["preprocess_data"]["simplify_news_path"]
pretrain_raw_path = config["preprocess_data"]["pretrain_raw_path"]
train_file_path = config["preprocess_data"]["train_file_path"]

# Changed absolute path to relative path
processed_data_path = "./data/fact_data/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = config['preprocess']['batch_size']

MAX_CLICK_LENGTH = config["preprocess"]["max_click_length"]
# LIMIT = config["preprocess"]["limit"]
LIMIT = 1
LLM_ENHANCE = True

# Create directory for processed data if it doesn't exist
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)
    # Translated Chinese log message
    logging.info(f"Created directory: {processed_data_path}")

# Initialize logging
log_filename = f"./logs/build_fact_datasets.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pretrain_data():
    """Loads all pretraining data."""
    # Translated Chinese log message
    logging.info("Loading data")
    news_corpus = pd.read_feather(pretrain_raw_path)
    #     news_id                                           o_titles                                              bodys
    # 0  N10000  Predicting Atlanta United's lineup against Col...  Only FIVE internationals allowed, count em, FI...
    ids = news_corpus["news_id"].tolist()
    titles = news_corpus["o_titles"].tolist()
    bodys = news_corpus["bodys"].tolist()

    assert len(ids) == len(titles) == len(bodys)
    # Translated Chinese log message
    logging.info("Data loaded successfully")
    return ids, titles, bodys

def save_high_confidence_samples(ids, titles, bodys, fact_scores, threshold=0.999):
    """Saves texts with high fact scores."""
    # Translated Chinese log message
    logging.info("Saving high confidence samples")
    ids_fact = pd.DataFrame({'news_id': ids, 'o_titles': titles, 'bodys': bodys, 'fact_scores': fact_scores})
    ids_fact.to_feather(processed_data_path + "fact_corpus.feather")

    selected = ids_fact[ids_fact['fact_scores'] > threshold]
    selected_ids = selected["news_id"].to_list()
    with open(processed_data_path + "selected_ids.pkl", "wb") as f:
        pickle.dump(selected_ids, f)

    # Translated Chinese log and print messages
    logging.info(f"High confidence samples saved. Total samples: {len(selected)}")
    print(f"High confidence samples saved. Total samples: {len(selected)}")
    return selected

def align_ws(old_token, new_token):
    """Aligns whitespace between old and new tokens."""
    if old_token[-1] == new_token[-1] == " ":
        return new_token
    elif old_token[-1] == " ":
        return new_token + " "
    elif new_token[-1] == " ":
        return new_token[:-1]
    else:
        return new_token

def swap_entities(text, claim, op):
    """Swaps entities in the text based on categories."""
    categories_map = [
        ("PERSON", "ORG", "NORP", "FAC", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT"),
        ("PERCENT", "MONEY", "QUANTITY", "CARDINAL"),
        ("DATE", "TIME")
    ]
    categories = categories_map[op]
    text_ents = [ent for ent in text.ents if ent.label_ in categories]
    claim_ents = [ent for ent in claim.ents if ent.label_ in categories]

    if not claim_ents or not text_ents:
        return None

    replaced_ent = random.choice(claim_ents)
    candidate_ents = [ent for ent in text_ents if ent.text != replaced_ent.text and ent.text not in replaced_ent.text and replaced_ent.text not in ent.text]

    if not candidate_ents:
        return None

    # python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')
    swapped_ent = random.choice(candidate_ents)
    claim_tokens = [token.text_with_ws for token in claim]
    swapped_token = align_ws(replaced_ent.text_with_ws, swapped_ent.text_with_ws)
    claim_swapped = claim_tokens[:replaced_ent.start] + [swapped_token] + claim_tokens[replaced_ent.end:]
    new_claim = nlp("".join(claim_swapped))
    if new_claim.text == claim.text:
        return None
    else:
        return new_claim.text

def process_negative_sample(sample):
    """Processes a single sample to generate a negative example by swapping entities."""
    ids, title, body, _ = sample
    nlp = spacy.load('en_core_web_sm')
    claim_docs = nlp(title)
    context_docs = nlp(body)
    order = [0, 1, 2]
    random.shuffle(order)
    for op in order:
        new_claim = swap_entities(context_docs, claim_docs, op)
        if new_claim is not None:
            return new_claim
    return None

def generate_negative_samples(selected, num_workers = 32):
    """Generates negative samples by swapping entities."""
    # Translated Chinese log message
    logging.info("Generating negative samples")
    samples = [tuple(selected.iloc[i]) for i in range(len(selected))]

    with Pool(num_workers) as pool:
        negative_examples = list(tqdm(pool.imap(process_negative_sample, samples), total=len(samples)))

    selected['neg_titles'] = negative_examples
    new_selected = selected[selected['neg_titles'].notnull()].reset_index(drop=True)
    new_selected.to_feather(processed_data_path + 'selected_corpus.feather')
    # Translated Chinese log and print messages
    logging.info(f"Negative samples generated. Total samples: {len(new_selected)}")
    print(f"Negative samples generated. Total samples: {len(new_selected)}")
    return new_selected

def build_cl_raw(data: pd.DataFrame, news_titles, news_bodys, news_dict, cl_news_samples, max_click_length=50, limit=None, dataset_type="train") -> Optional[pd.DataFrame]:
    # Translated Chinese print message
    print(f"Starting to build {dataset_type}_raw dataset")
    selected_ids = cl_news_samples['news_id'].tolist()
    selected_pos = cl_news_samples['o_titles'].tolist()
    selected_neg = cl_news_samples['neg_titles'].tolist()
    selected_dict = {}
    index = 1
    for id in selected_ids:
        selected_dict[id] = index
        index += 1

    history_inputs, history_ids_list, bodys, ids = [], [], [], []
    pos_titles, neg_titles= [], []
    user_ids_list = []  # List to store user IDs
    news_count, user_count = {}, {}

    # Iterate through dataset rows
    for user_id, click_history_ids, _, _, pos_ids, neg_ids, _, _, _ in tqdm(data.itertuples(index=False), total=len(data)):
        # Process click history to get history and history_ids
        click_history, history_ids_mapped = process_click_history(click_history_ids, news_titles, news_dict, max_click_length, data_type=dataset_type)
        pos_ids = pos_ids.split(" ")

        for pos_id in pos_ids:
            if pos_id not in selected_ids:
                continue

            # Use get method to prevent KeyError
            body = news_bodys.get(news_dict.get(pos_id, news_dict['UNK']), "")
            pos_title = selected_pos[selected_dict.get(pos_id, -1) - 1] if selected_dict.get(pos_id, -1) > 0 else ""
            neg_title = selected_neg[selected_dict.get(pos_id, -1) - 1] if selected_dict.get(pos_id, -1) > 0 else ""

            # Skip empty titles or bodies
            if not pos_title.strip() or not neg_title.strip() or not body.strip():
                continue

            # If limit exists, check if limit is reached, skip news if so
            if limit and news_count.get(pos_id, 0) >= limit:
                continue

            # Update news count, initialize to 0 if news ID doesn't exist
            news_count[pos_id] = news_count.get(pos_id, 0) + 1

            # Update user count
            user_count[user_id] = user_count.get(user_id, 0) + 1

            # Add data to lists
            ids.append(pos_id)
            history_inputs.append(click_history)
            history_ids_list.append(history_ids_mapped)  # Add history_ids
            user_ids_list.append(user_id)  # Add user ID to list
            bodys.append(body)
            pos_titles.append(pos_title)
            neg_titles.append(neg_title)

    # Build DataFrame
    dataset_raw_samples = pd.DataFrame({
        'history': history_inputs,
        'history_ids': history_ids_list,  # Add history_ids column
        'user_id': user_ids_list,  # Include user ID
        'news_id': ids,
        'bodys': bodys,
        'pos_titles': pos_titles,
        'neg_titles': neg_titles
    })

    # Save dataset as feather file
    # Changed absolute path to relative path
    base_name = f"./data/fact_data/{dataset_type}_raw"
    dataset_raw_path = base_name + f"_limit_{limit}.feather" if limit else base_name + "_all.feather"
    dataset_raw_samples.to_feather(dataset_raw_path)

    train_ids = list(news_count.keys())
    user_ids = list(user_count.keys())
    sam_per_user = list(user_count.values())

    # Translated Chinese log and print messages
    logging.info("Contrastive learning data processing completed.")
    logging.info(f"Number of news: {len(train_ids)}")
    logging.info(f"Number of users: {len(user_ids)}")
    logging.info(f"Number of samples: {len(dataset_raw_samples)}")
    logging.info(f"Average samples per user: {np.mean(sam_per_user)}")
    logging.info(f"Maximum samples per user: {np.max(sam_per_user)}")

    print("Contrastive learning data processing completed.")
    print(f"Number of news: {len(train_ids)}")
    print(f"Number of users: {len(user_ids)}")
    print(f"Number of samples: {len(dataset_raw_samples)}")
    print(f"Average samples per user: {np.mean(sam_per_user)}")
    print(f"Maximum samples per user: {np.max(sam_per_user)}")

    print(dataset_raw_samples.head())

    return dataset_raw_samples

def clean_high_score_negative_samples(cl_news_samples, threshold=0.9):
    """Cleans negative samples with high fact scores."""
    # Translated Chinese log message
    logging.info("Cleaning high score negative samples")
    # Calculate scores for neg_titles and bodys
    neg_scores = fact_score(hyps= cl_news_samples['neg_titles'].tolist(), bodys=cl_news_samples['bodys'].tolist(), device=device)
    # Add score column
    cl_news_samples['neg_scores'] = neg_scores
    # Keep samples with scores below the threshold, print number of removed samples
    removed = cl_news_samples[cl_news_samples['neg_scores'] >= threshold]
    saved = cl_news_samples[cl_news_samples['neg_scores'] < threshold].reset_index(drop=True)
    # Translated Chinese log and print messages
    logging.info(f"High score negative samples cleaned. Removed samples: {len(removed)}")
    print(f"all samples: {len(cl_news_samples)}")
    print(f"High score negative samples cleaned. Removed samples: {len(removed)}")
    print(f"Saved samples: {len(saved)}")
    return saved

def enhance_negative_samples(cl_news_samples, batch_size=16):
    """Enhances negative samples using an LLM."""
    from vllm import LLM, SamplingParams

    # Note: This path might need adjustment based on actual model location relative to the script.
    llm = LLM(model=LLM_PATH, dtype="bfloat16", max_model_len = 4096)
    sampling_params = SamplingParams(top_p=0.9, temperature=0.8, max_tokens=256, min_tokens=5)

    logging.info("Starting enhancement of negative samples using LLM.")
    # Translated Chinese print message
    print("Enhancing negative samples using LLM...")
    # Total number of samples
    total_samples = len(cl_news_samples)
    logging.info(f"Total samples to enhance: {total_samples}")
    print(f"Total samples to enhance: {total_samples}")

    enhanced_neg_titles = []
    enhanced_indices = []  # Record indices of successfully processed samples

    # Process in batches
    for start_idx in tqdm(range(0, total_samples, batch_size), desc="Enhancing batches"):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_neg_titles = cl_news_samples['neg_titles'].iloc[start_idx:end_idx].tolist()
        batch_indices = list(range(start_idx, end_idx))  # Indices for the current batch

        # Prepare conversations for the current batch
        conversations = []
        for neg_title in batch_neg_titles:
            conversation = [
                {
                "role": "system",
                "content": "You are a helpful assistant."
                },
                {
                "role": "user",
                "content": f"Please rewrite the following news headlines, and only return the results without any extra symbols or explanations.\n{neg_title}"
                }
            ]
            conversations.append(conversation)

        # Perform batch inference
        try:
            outputs = llm.chat(
                messages=conversations,
                sampling_params=sampling_params,
                use_tqdm=False  # Disable inner tqdm since we're already using an outer one
            )
        except Exception as e:
            logging.error(f"Error during batch {start_idx // batch_size + 1} inference: {e}")
            print(f"Error during batch {start_idx // batch_size + 1} inference: {e}")
            continue

        # Extract enhanced 'neg_titles' from outputs
        for idx, output in enumerate(outputs):
            if output.outputs and len(output.outputs) > 0:
                enhanced_text = output.outputs[0].text.strip()
                enhanced_neg_titles.append(enhanced_text)
                enhanced_indices.append(batch_indices[idx])  # Save successfully processed index
            else:
                logging.warning(f"No output for batch {start_idx // batch_size + 1}, sample index {idx}. Skipping this sample.")
                # Skip this sample

    logging.info("Extraction of enhanced 'neg_titles' completed.")
    print("Extraction of enhanced 'neg_titles' completed.")

    # Create a new DataFrame with enhanced samples
    enhanced_samples = cl_news_samples.copy()
    enhanced_samples['neg_titles'] = enhanced_neg_titles

    # Concatenate the original and enhanced DataFrames
    # Note: This concatenation strategy might duplicate samples if LLM enhancement is successful.
    # A better approach might be to update the 'neg_titles' column in the original DataFrame
    # only for the successfully enhanced samples, or create a new DataFrame with original
    # samples and enhanced samples clearly distinguished if needed.
    # For now, following the original code's logic of concatenating.
    cl_news_samples_enhanced = pd.concat([cl_news_samples, enhanced_samples], ignore_index=True)

    return cl_news_samples_enhanced

def main():
    logging.info("Starting process")
    # Changed absolute path to relative path
    file_path = "./data/fact_data/" + "selected_corpus.feather"
    if os.path.exists(file_path):
        # Translated Chinese print message
        print("Found existing file")
        cl_news_samples = pd.read_feather(file_path)
    else:
        # Load news data
        # Translated Chinese comment
        logging.info("Loading pretrain data as selected_corpus.feather not found.")
        ids, titles, bodys = load_pretrain_data()

        fact_scores = fact_score(titles, bodys, device)
        logging.info(f"Prediction completed. Average score: {np.mean(fact_scores)}")
        print(f"Prediction completed. Average score: {np.mean(fact_scores)}")

        selected_samples = save_high_confidence_samples(ids, titles, bodys, fact_scores)
        # Translated Chinese print message
        print("Save completed")

        cl_news_samples = generate_negative_samples(selected_samples)
        # Translated Chinese print message
        print("Generate completed")

    print(cl_news_samples.head())

    news = load_feather_data(simplify_news_path)

    # Build news ID to index mapping table
    # Translated Chinese comment
    news_ids = news["News ID"].unique().tolist()
    news_dict = build_news_dict(news_ids)

    # Store news titles and bodies as dictionaries for fast indexing
    # Ensure keys are consistent with indices in news_dict
    # Translated Chinese comment
    news_titles = {idx: title for idx, title in enumerate(news["Headline"].values, start=1)}
    news_bodys = {idx: body for idx, body in enumerate(news["News body"].values, start=1)}

    # Process contrastive learning data
    # Translated Chinese print message
    print("Processing contrastive learning data")
    train = load_tsv_data(train_file_path)
    build_cl = build_cl_raw(train, news_titles, news_bodys, news_dict, cl_news_samples, max_click_length=MAX_CLICK_LENGTH, limit=LIMIT, dataset_type="train_cl")
    if LLM_ENHANCE:
        build_cl_llm = enhance_negative_samples(build_cl)
        # Changed absolute path to relative path
        cl_llm_path = "./data/fact_data/" + f"train_cl_raw_limit_{LIMIT}_llm.feather" if LIMIT else "./data/fact_data/" + "_all_llm.feather"
        build_cl = clean_high_score_negative_samples(build_cl_llm)
        build_cl.to_feather(cl_llm_path)

    else:
        # Changed absolute path to relative path
        cl_path = "./data/fact_data/" + f"train_cl_raw_limit_{LIMIT}.feather" if LIMIT else "./data/fact_data/" + "_all.feather"
        build_cl = clean_high_score_negative_samples(build_cl)
        build_cl.to_feather(cl_path)

    print(build_cl)

    # Calculate new metrics
    # Translated Chinese comment
    news_count = build_cl['news_id'].nunique()
    user_count = build_cl['user_id'].nunique()
    sample_count = len(build_cl)
    samples_per_user = build_cl.groupby('user_id').size()
    average_samples_per_user = samples_per_user.mean()
    max_samples_per_user = samples_per_user.max()

    # Translated Chinese log and print messages
    logging.info(f"Number of news: {news_count}")
    logging.info(f"Number of users: {user_count}")
    logging.info(f"Number of samples: {sample_count}")
    logging.info(f"Average samples per user: {average_samples_per_user}")
    logging.info(f"Maximum samples per user: {max_samples_per_user}")

    print(f"Number of news: {news_count}")
    print(f"Number of users: {user_count}")
    print(f"Number of samples: {sample_count}")
    print(f"Average samples per user: {average_samples_per_user}")
    print(f"Maximum samples per user: {max_samples_per_user}")

    logging.info("Process completed successfully")

if __name__ == "__main__":
    main()