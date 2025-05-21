
from functools import partial
import logging
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import yaml

# Load configuration from path_config.yaml
# Changed absolute path to relative path
with open("./path_config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Get max lengths from config
max_length = config['preprocess']['max_length']
max_title_length = config['preprocess']['max_title_length']
max_click_length = config['preprocess']['max_click_length']

def collect_data(batch, pretrain=False):
        """
        Collate function for DataLoader. Converts lists of tensors/data in a batch
        into batched tensors and lists.

        Args:
            batch (list): A list of data samples.
            pretrain (bool): Flag to indicate if it's for pretraining data.

        Returns:
            dict: A dictionary containing batched tensors and lists.
        """
        # Convert lists to tensors
        body_inputs = torch.tensor([example['input_ids'] for example in batch], dtype=torch.long)
        body_masks = torch.tensor([example['attention_mask'] for example in batch], dtype=torch.long)
        labels = torch.tensor([example['labels'] for example in batch], dtype=torch.long)

        data = {
            'input_ids': body_inputs,
            'attention_mask': body_masks,
            'labels': labels
        }
        if not pretrain:
            # Convert h_inputs and h_masks to tensors
            h_inputs = torch.tensor([example['h_inputs'] for example in batch], dtype=torch.long)  # Shape: [batch_size, max_click_length]
            h_masks = torch.tensor([example['h_masks'] for example in batch], dtype=torch.long)    # Shape: [batch_size, max_click_length]

            data['h_inputs'] = h_inputs
            data['h_masks'] = h_masks

            data['bodys'] = [example['bodys'] for example in batch]
            data['history'] = [example['history'] for example in batch]

        return data

def collect_train_data(batch):
        """
        Collate function specifically for training data DataLoader.
        Converts lists of tensors/data in a batch into batched tensors and lists.

        Args:
            batch (list): A list of training data samples.

        Returns:
            dict: A dictionary containing batched tensors and lists for training.
        """
        # Convert lists to tensors
        body_inputs = torch.tensor([example['input_ids'] for example in batch], dtype=torch.long)
        body_masks = torch.tensor([example['attention_mask'] for example in batch], dtype=torch.long)
        labels = torch.tensor([example['labels'] for example in batch], dtype=torch.long)

        # Convert h_inputs and h_masks to tensors
        h_inputs = torch.tensor([example['h_inputs'] for example in batch], dtype=torch.long)  # Shape: [batch_size, max_click_length]
        h_masks = torch.tensor([example['h_masks'] for example in batch], dtype=torch.long)    # Shape: [batch_size, max_click_length]

        data = {
            'input_ids': body_inputs,
            'attention_mask': body_masks,
            'labels': labels,
            'h_inputs': h_inputs,
            'h_masks': h_masks
        }

        data['bodys'] = [example['bodys'] for example in batch]
        data['history'] = [example['history'] for example in batch]

        return data

def pretrain_data(tokenizer, batch_size=6, test_batch_size=16):
    """
    Returns data loaders for pretraining and validation datasets.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        batch_size (int): Batch size for the pretraining DataLoader.
        test_batch_size (int): Batch size for the validation DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the pretraining DataLoader
                                       and the validation DataLoader.
    """

    def tokenize_pretrain(examples):
        """
        Tokenizes pretraining data examples.

        Args:
            examples (dict): A dictionary containing 'bodys' and 'o_titles'.

        Returns:
            dict: A dictionary containing tokenized inputs and labels.
        """
        inputs = examples['bodys']
        targets = examples['o_titles']

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,  # Max length for input text
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            targets,
            max_length=max_title_length,  # Max length for target text
            truncation=True,
            padding="max_length",
        )

        # Replace pad_token_id with -100 for language modeling loss calculation
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label]
            for label in labels_ids
        ]

        model_inputs["labels"] = labels_ids  # Use the modified labels
        return model_inputs

    # Load pretraining data and tokenize
    pretrain_df = pd.read_feather(config['preprocess_data']['pretrain_raw_ex_path'])
    pretrain_dataset = Dataset.from_pandas(pretrain_df)
    pretrain_dataset = pretrain_dataset.map(tokenize_pretrain, batched=True, num_proc=8, remove_columns=['__index_level_0__', 'news_id', 'o_titles', 'bodys'])
    # Expected features after mapping: ['input_ids', 'attention_mask', 'labels']

    # Create DataLoader for pretraining
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=partial(collect_data, pretrain=True))

    # Get validation DataLoader
    test_dataloader = test_data(tokenizer, batch_size=test_batch_size)

    logging.info("DataLoaders for pretraining and validation datasets created successfully.")

    return pretrain_dataloader, test_dataloader

def test_data(tokenizer, batch_size=16):
    """
    Returns a DataLoader for the test dataset.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        batch_size (int): Batch size for the test DataLoader.

    Returns:
        DataLoader: The test DataLoader.
    """
    dataset = test_dataset(tokenizer)
    test_dataloader = DataLoader(dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=partial(collect_data, pretrain=False))

    return test_dataloader

def test_data_save(tokenizer, batch_size=16):
    """
    Returns a DataLoader for the test dataset, including user_id and news_id for saving results.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        batch_size (int): Batch size for the test DataLoader.

    Returns:
        DataLoader: The test DataLoader including user and news IDs.
    """
    def collect_data_save(batch):
        """
        Collate function for test data DataLoader when saving results.
        Includes user_id and news_id in the output batch.

        Args:
            batch (list): A list of test data samples.

        Returns:
            dict: A dictionary containing batched tensors, lists, user_ids, and news_ids.
        """
        # Convert lists to tensors
        body_inputs = torch.tensor([example['input_ids'] for example in batch], dtype=torch.long)
        body_masks = torch.tensor([example['attention_mask'] for example in batch], dtype=torch.long)
        labels = torch.tensor([example['labels'] for example in batch], dtype=torch.long)

        data = {
            'input_ids': body_inputs,
            'attention_mask': body_masks,
            'labels': labels
        }
        # Convert h_inputs and h_masks to tensors
        h_inputs = torch.tensor([example['h_inputs'] for example in batch], dtype=torch.long)  # Shape: [batch_size, max_click_length]
        h_masks = torch.tensor([example['h_masks'] for example in batch], dtype=torch.long)    # Shape: [batch_size, max_click_length]

        data['h_inputs'] = h_inputs
        data['h_masks'] = h_masks

        data['bodys'] = [example['bodys'] for example in batch]
        data['history'] = [example['history'] for example in batch]
        data['user_id'] = [example['user_id'] for example in batch]
        data['news_id'] = [example['news_id'] for example in batch]

        return data

    dataset = test_dataset(tokenizer, save = True)
    test_dataloader = DataLoader(dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=partial(collect_data_save))

    return test_dataloader

def training_data(tokenizer, train_batch_size=4, test_batch_size=16):
    """
    Returns data loaders for the training and validation datasets for the user history encoder.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        train_batch_size (int): Batch size for the training DataLoader.
        test_batch_size (int): Batch size for the validation DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader
                                       and the validation DataLoader.
    """
    train_dataset = training_dataset(tokenizer)

    # Expected features after mapping: ['history', 'bodys', 'input_ids', 'attention_mask', 'labels', 'h_inputs', 'h_masks']
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, collate_fn=partial(collect_train_data))

    test_dataloader = test_data(tokenizer,batch_size=test_batch_size)

    logging.info("DataLoaders for train and validation datasets created successfully.")

    return train_dataloader, test_dataloader

def training_dataset(tokenizer):
    """
    Prepares the training dataset for the user history encoder.

    Args:
        tokenizer: The tokenizer to use for tokenization.

    Returns:
        Dataset: The processed training dataset.
    """
    def tokenize_train(examples):
        """
        Tokenizes training data examples and processes history IDs.

        Args:
            examples (dict): A dictionary containing 'bodys', 'o_titles', and 'history_ids'.

        Returns:
            dict: A dictionary containing tokenized inputs, labels, and processed history data.
        """
        # Expected input features: ['history', 'history_ids', 'user_id', 'news_id', 'bodys', 'o_titles', 'neg_titles']
        inputs = examples['bodys']
        targets = examples['o_titles']
        history_ids = examples['history_ids']

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,  # Max length for input text
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            targets,
            max_length=max_title_length,  # Max length for target text
            truncation=True,
            padding="max_length",
        )

        # Replace pad_token_id with -100 for language modeling loss calculation
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label]
            for label in labels_ids
        ]

        model_inputs["labels"] = labels_ids  # Use the modified labels

        # Process history_ids list
        h_inputs = []
        h_masks = []
        for history in history_ids:
            if len(history) < max_click_length:
                padding_length = max_click_length - len(history)
                padded_history = history + [0] * padding_length  # Assuming 0 is PAD token ID
                h_mask = [1] * len(history) + [0] * padding_length  # 1 for actual data, 0 for padding
            else:
                padded_history = history[-max_click_length:]
                h_mask = [1] * max_click_length  # All 1s for truncated history

            h_inputs.append(padded_history)
            h_masks.append(h_mask)

        model_inputs["h_inputs"] = h_inputs  # Add processed history token IDs
        model_inputs["h_masks"] = h_masks        # Add corresponding mask

        return model_inputs

    # Load training data and tokenize
    train_df = pd.read_feather(config['data']['train_data_path'])
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(tokenize_train, batched=True, num_proc=8, remove_columns=['user_id', 'news_id', 'history_ids', 'o_titles', 'neg_titles'])
    # Expected features after mapping: ['history', 'bodys', 'input_ids', 'attention_mask', 'labels', 'h_inputs', 'h_masks']

    return train_dataset

def test_dataset(tokenizer, save = False):
    """
    Prepares the test dataset.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        save (bool): Flag to indicate if user_id and news_id should be kept for saving results.

    Returns:
        Dataset: The processed test dataset.
    """

    def tokenize_test(examples):
        """
        Tokenizes test data examples and processes history IDs.

        Args:
            examples (dict): A dictionary containing 'bodys', 'p_titles', and 'history_ids'.

        Returns:
            dict: A dictionary containing tokenized inputs, labels, and processed history data.
        """
        inputs = examples['bodys']
        targets = examples['p_titles']
        history_ids = examples['history_ids']

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,  # Max length for input text
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            targets,
            max_length=max_title_length,  # Max length for target text
            truncation=True,
            padding="max_length",
        )

        # Replace pad_token_id with -100 for language modeling loss calculation
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label]
            for label in labels_ids
        ]

        model_inputs["labels"] = labels_ids  # Use the modified labels
        # Process history_ids list
        h_inputs = []
        h_masks = []
        for history in history_ids:
            if len(history) < max_click_length:
                padding_length = max_click_length - len(history)
                padded_history = history + [0] * padding_length  # Assuming 0 is PAD token ID
                h_mask = [1] * len(history) + [0] * padding_length  # 1 for actual data, 0 for padding
            else:
                padded_history = history[-max_click_length:]
                h_mask = [1] * max_click_length  # All 1s for truncated history

            h_inputs.append(padded_history)
            h_masks.append(h_mask)

        model_inputs["h_inputs"] = h_inputs  # Add processed history token IDs
        model_inputs["h_masks"] = h_masks        # Add corresponding mask

        return model_inputs

    # Load test data and tokenize
    test_df = pd.read_feather(config['preprocess_data']['test_raw_path'])
    test_dataset = Dataset.from_pandas(test_df)
    if save:
        # Keep user_id and news_id columns if save is True
        test_dataset = test_dataset.map(tokenize_test, batched=True, num_proc=8, remove_columns=['history_ids'])
    else:
        # Remove user_id and news_id columns if save is False
        test_dataset = test_dataset.map(tokenize_test, batched=True, num_proc=8, remove_columns=['user_id', 'news_id', 'history_ids'])
    # Expected features after mapping: ['history', 'bodys', 'o_titles', 'p_titles', 'input_ids', 'attention_mask', 'labels', 'h_inputs', 'h_masks'] (plus user_id, news_id if save=True)

    return test_dataset

def perference_dataset(tokenizer):
    """
    Prepares the preference dataset for DPO training.

    Args:
        tokenizer: The tokenizer to use for tokenization.

    Returns:
        Dataset: The processed preference dataset.
    """
    def tokenize_dpo(examples):
        """
        Tokenizes preference data examples (for DPO) and processes history IDs.

        Args:
            examples (dict): A dictionary containing 'bodys', 'pos_titles', 'neg_titles', and 'history_ids'.

        Returns:
            dict: A dictionary containing tokenized inputs, positive/negative labels, and processed history data.
        """
        # Expected input features: ['history', 'history_ids', 'user_id', 'news_id', 'bodys', 'pos_titles', 'neg_titles']
        inputs = examples['bodys']
        targets = examples['pos_titles']
        neg_targets = examples['neg_titles']
        history_ids = examples['history_ids']

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,  # Max length for input text
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            targets,
            max_length=max_title_length,  # Max length for target text
            truncation=True,
            padding="max_length",
        )
        neg_labels = tokenizer(
            neg_targets,
            max_length=max_title_length,  # Max length for target text
            truncation=True,
            padding="max_length",
        )

        # Replace pad_token_id with -100 for language modeling loss calculation
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label]
            for label in labels_ids
        ]
        neg_labels_ids = neg_labels["input_ids"]
        neg_labels_ids = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label]
            for label in neg_labels_ids
        ]

        model_inputs["labels"] = labels_ids  # Use the modified positive labels
        model_inputs["neg_labels"] = neg_labels_ids  # Use the modified negative labels

        # Process history_ids list
        h_inputs = []
        h_masks = []
        for history in history_ids:
            if len(history) < max_click_length:
                padding_length = max_click_length - len(history)
                padded_history = history + [0] * padding_length  # Assuming 0 is PAD token ID
                h_mask = [1] * len(history) + [0] * padding_length  # 1 for actual data, 0 for padding
            else:
                padded_history = history[-max_click_length:]
                h_mask = [1] * max_click_length  # All 1s for truncated history

            h_inputs.append(padded_history)
            h_masks.append(h_mask)

        model_inputs["h_inputs"] = h_inputs  # Add processed history token IDs
        model_inputs["h_masks"] = h_masks        # Add corresponding mask

        return model_inputs

    # Load preference data and tokenize
    dpo_df = pd.read_feather(config['data']['cl_data_path'])
    dpo_dataset = Dataset.from_pandas(dpo_df)
    dpo_dataset = dpo_dataset.map(tokenize_dpo, batched=True, num_proc=8, remove_columns=['user_id', 'news_id', 'history_ids', 'pos_titles', 'neg_titles'])
    # Expected features after mapping: ['history', 'bodys', 'input_ids', 'attention_mask', 'labels', 'neg_labels', 'h_inputs', 'h_masks']

    return dpo_dataset

def perference_data(tokenizer, train_batch_size=4):
    """
    Returns a DataLoader for the preference dataset (for DPO training).

    Args:
        tokenizer: The tokenizer to use for tokenization.
        train_batch_size (int): Batch size for the preference DataLoader.

    Returns:
        DataLoader: The preference DataLoader.
    """
    def collect_perference_data(batch):
        """
        Collate function for preference data DataLoader.
        Converts lists of tensors/data in a batch into batched tensors and lists.

        Args:
            batch (list): A list of preference data samples.

        Returns:
            dict: A dictionary containing batched tensors and lists for preference training.
        """
        # Convert lists to tensors
        body_inputs = torch.tensor([example['input_ids'] for example in batch], dtype=torch.long)
        body_masks = torch.tensor([example['attention_mask'] for example in batch], dtype=torch.long)
        labels = torch.tensor([example['labels'] for example in batch], dtype=torch.long)
        neg_labels = torch.tensor([example['neg_labels'] for example in batch], dtype=torch.long)

        # Convert h_inputs and h_masks to tensors
        h_inputs = torch.tensor([example['h_inputs'] for example in batch], dtype=torch.long)  # Shape: [batch_size, max_click_length]
        h_masks = torch.tensor([example['h_masks'] for example in batch], dtype=torch.long)    # Shape: [batch_size, max_click_length]

        data = {
            'input_ids': body_inputs,
            'attention_mask': body_masks,
            'labels': labels,
            'neg_labels': neg_labels,
            'h_inputs': h_inputs,
            'h_masks': h_masks
        }

        data['bodys'] = [example['bodys'] for example in batch]
        data['history'] = [example['history'] for example in batch]

        return data

    dataset = perference_dataset(tokenizer)

    # Expected features after mapping: ['history', 'bodys', 'input_ids', 'attention_mask', 'labels', 'neg_labels', 'h_inputs', 'h_masks']
    perference_dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, collate_fn=partial(collect_perference_data))

    logging.info("DataLoaders for preference datasets created successfully.")

    return perference_dataloader
