from multiprocessing import Pool
import os
import numpy as np
import evaluate  # Use Hugging Face's evaluate library
from simtext import similarity
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import yaml # Import yaml to load config

import jieba
jieba.setLogLevel(jieba.logging.INFO)

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load configuration file
# Use a relative path to the config file
with open("./path_config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Model and device settings
# Load fact model path from config
fact_model_path = config['model']['fact_model_path']


def rouge_score(hyps, refs, use_stemmer=False):
    """
    Calculates ROUGE scores using Hugging Face's evaluate library.

    Args:
        hyps (List[str]): Generated summaries (hypotheses).
        refs (List[str]): Reference summaries.
        use_stemmer (bool): Whether to use stemming for ROUGE calculation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ROUGE-1, ROUGE-2, ROUGE-L scores.
    """
    # Initialize ROUGE evaluator
    rouge_evaluator = evaluate.load("rouge")

    scores = rouge_evaluator.compute(predictions=hyps, references=refs, use_stemmer=use_stemmer)

    rouge_1 = np.array(scores['rouge1'])
    rouge_2 = np.array(scores['rouge2'])
    rouge_l = np.array(scores['rougeL'])

    return rouge_1, rouge_2, rouge_l

def fact_score(hyps, bodys, device, batch_size=32):
    """
    Calculates FactCC scores for generated summaries against source bodies.

    Args:
        hyps (List[str]): Generated summaries (hypotheses).
        bodys (List[str]): Source document bodies.
        device (torch.device): Device to run the model on (e.g., 'cuda', 'cpu').
        batch_size (int): Batch size for model inference.

    Returns:
        List[float]: List of fact scores (probability of being factual) for each hypothesis.
    """
    fact_scores = []

    # FactCC model initialization
    fact_tokenizer = BertTokenizer.from_pretrained(fact_model_path)
    # Ensure local_files_only is True if the model is expected to be local
    fact_model = BertForSequenceClassification.from_pretrained(fact_model_path, local_files_only=True, torch_dtype=torch.float16).to(device)
    fact_model.eval()

    # Create dataset from hypotheses and bodies
    dataset = Dataset.from_dict({'hyps': hyps, 'bodys': bodys})

    # Define tokenization function for batch processing
    def tokenization(examples):
        # Batch tokenization of hyps and bodys
        tokenized_inputs = fact_tokenizer(examples['bodys'], examples['hyps'], padding='max_length', truncation='only_first', max_length=512, return_tensors='pt')
        return tokenized_inputs

    # Use map function for batch tokenization
    tokenized_dataset = dataset.map(tokenization, batched=True, num_proc=16)

    # Define collate function for DataLoader
    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        token_type_ids = torch.tensor([item['token_type_ids'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

    # Create DataLoader for batch processing
    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, num_workers=16,collate_fn=collate_fn)


    with torch.no_grad():
        for batch in tqdm(data_loader, ncols=100, desc="Calculating FactCC scores"):
            inputs = {k: v.to(device) for k, v in batch.items()}

            logits = fact_model(**inputs).logits
            # Get probability of the factual class (assuming class 0 is factual)
            probs = F.softmax(logits, dim=1).to(torch.float32)
            fact_scores.extend(probs[:, 0].cpu().numpy().tolist())

    return fact_scores


def compute_scores_for_one_history(hyp, history):
    """
    Computes Jaccard and Cosine similarity scores for a hypothesis against a history of texts.

    Args:
        hyp (str): The generated hypothesis.
        history (List[str]): A list of historical texts.

    Returns:
        Tuple[float, float, float, float]: Mean Jaccard, Max Jaccard, Mean Cosine, Max Cosine scores.
    """
    sim = similarity()
    j_scores = []
    c_scores = []
    for h in history:
        # Compute similarity scores between history item and hypothesis (lowercase)
        sim_scores = sim.compute(h.lower(), hyp.lower())
        c_scores.append(sim_scores["Sim_Cosine"])
        j_scores.append(sim_scores['Sim_Jaccard'])
    # Return mean and max scores
    return np.mean(j_scores), np.max(j_scores), np.mean(c_scores), np.max(c_scores)

def personal_score(hyps, test_history, num_workers = 32):
    """
    Calculates personalization scores (Jaccard and Cosine similarity) for hypotheses against user history.

    Args:
        hyps (List[str]): Generated summaries (hypotheses).
        test_history (List[List[str]]): List of user history lists.
        num_workers (int): Number of worker processes for parallel computation.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            Lists of mean Jaccard, max Jaccard, mean Cosine, and max Cosine scores.
    """
    p_j_mean_scores = []
    p_j_max_scores = []
    p_c_mean_scores = []
    p_c_max_scores = []

    # Use multiprocessing Pool for parallel computation
    with Pool(num_workers) as pool:
        results = []
        # Apply compute_scores_for_one_history asynchronously for each hypothesis and its history
        for i in range(len(test_history)):
            results.append(pool.apply_async(compute_scores_for_one_history, (hyps[i], test_history[i])))

        # Collect results from the pool
        for r in tqdm(results, ncols=100, desc="Calculating Personal scores"):
            j_mean, j_max, c_mean, c_max = r.get()
            p_j_mean_scores.append(j_mean)
            p_j_max_scores.append(j_max)
            p_c_mean_scores.append(c_mean)
            p_c_max_scores.append(c_max)

    return p_j_mean_scores, p_j_max_scores, p_c_mean_scores, p_c_max_scores

def all_metrics(refs, hyps, test_bodys, test_history, device, rouge_only=False):
    """
    Calculates ROUGE, FactCC, and Personalization scores for generated summaries.

    Args:
        refs (List[str]): Reference summaries.
        hyps (List[str]): Generated summaries (hypotheses).
        test_bodys (List[str]): Source document bodies.
        test_history (List[List[str]]): List of user history lists.
        device (torch.device): Device to run the FactCC model on.
        rouge_only (bool): If True, only calculate ROUGE scores.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing arrays of all calculated scores.
    """

    assert len(refs) == len(hyps) == len(test_bodys) == len(test_history), "Input lists must have the same length."

    # Helper function to convert data to lowercase recursively
    def to_lowercase(data):
        if isinstance(data, list):
            return [to_lowercase(item) for item in data]
        return str(data).lower()

    # Convert all input data to lowercase for consistent comparison
    refs_lower = to_lowercase(refs)
    hyps_lower = to_lowercase(hyps)
    test_bodys_lower = to_lowercase(test_bodys)
    test_history_lower = to_lowercase(test_history)

    # Calculate ROUGE scores
    rouge_1, rouge_2, rouge_l = rouge_score(hyps_lower, refs_lower, use_stemmer=True)

    if rouge_only:
        # Print ROUGE scores if only ROUGE is requested
        print('[Test]: ROUGE-1 Score: {:.4f}, ROUGE-2 Score: {:.4f}, ROUGE-L Score: {:.4f}'.format(100 * rouge_1.mean(),
                                                                                                100 * rouge_2.mean(), 100 * rouge_l.mean()))

        # Return scores dictionary with zeros for other metrics
        all_scores = {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'factcc': 0,
            'personal_j_mean': 0,
            'personal_j_max': 0,
            'personal_c_mean': 0,
            'personal_c_max': 0
        }
    else:
        # Calculate FactCC scores
        fact_scores = fact_score(hyps_lower, test_bodys_lower, device)

        # Calculate Personalization scores
        p_j_mean_scores, p_j_max_scores, p_c_mean_scores, p_c_max_scores = personal_score(hyps_lower, test_history_lower)

        # Print all calculated scores
        print('[Test]: ROUGE-1 Score: {:.4f}, ROUGE-2 Score: {:.4f}, ROUGE-L Score: {:.4f}'.format(100 * rouge_1.mean(),
                                                                                                100 * rouge_2.mean(), 100 * rouge_l.mean()))
        print('[Test]: FactCC Score: {:.4f}'.format(100 * np.mean(fact_scores)))
        print("[Test]: Personal-J-mean:{},\t Personal-J-max:{}".format(100 * np.mean(p_j_mean_scores), 100 * np.mean(p_j_max_scores)))
        print("[Test]: Personal-C-mean:{},\t Personal-C-max:{}".format(100 * np.mean(p_c_mean_scores), 100 * np.mean(p_c_max_scores)))

        # Return dictionary containing arrays of all scores (scaled by 100)
        all_scores = {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'factcc': 100 * np.array(fact_scores),
            'personal_j_mean': 100 * np.array(p_j_mean_scores),
            'personal_j_max': 100 * np.array(p_j_max_scores),
            'personal_c_mean': 100 * np.array(p_c_mean_scores),
            'personal_c_max': 100 * np.array(p_c_max_scores)
        }

    return all_scores

def main():
    """
    Main function to demonstrate metric calculation with sample data.
    Loads sample data, initializes device, and calls the all_metrics function.
    """
    # import argparse
    # import json

    # Optional: Uncomment the following lines if you want to parse arguments from the command line
    # parser = argparse.ArgumentParser(description="Evaluate generated summaries.")
    # parser.add_argument('--refs', type=str, required=True, help='Path to references file (JSON lines)')
    # parser.add_argument('--hyps', type=str, required=True, help='Path to hypotheses file (JSON lines)')
    # parser.add_argument('--test_bodys', type=str, required=True, help='Path to test bodies file (JSON lines)')
    # parser.add_argument('--test_history', type=str, required=True, help='Path to test history file (JSON lines)')
    # parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    # args = parser.parse_args()

    # For demonstration purposes, we'll use sample data. Replace this with actual data loading as needed.
    refs = [
        "The cat sat on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world."
    ]

    hyps = [
        "The cat is sitting on the mat.",
        "A fast brown fox leaps over a lazy dog.",
        "AI is changing the world."
    ]

    test_bodys = [
        "Cats are small, carnivorous mammals that are often kept as pets.",
        "Foxes are known for their quickness and agility.",
        "Artificial intelligence encompasses machine learning, deep learning, and more."
    ]

    test_history = [
        ["The cat plays with a ball.", "The cat is playful."],
        ["Foxes live in forests.", "Foxes are intelligent."],
        ["AI applications are widespread.", "AI impacts various industries."]
    ]

    # Initialize the device
    # Using cuda:4 if available, otherwise cpu
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Call the all_metrics function to compute and print scores
    all_scores = all_metrics(refs, hyps, test_bodys, test_history, device)

if __name__ == "__main__":
    main()
