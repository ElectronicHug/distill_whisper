from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from datasets import DatasetDict

def prepare_vectorized_dataset(data_args,
    model, feature_extractor, tokenizer, raw_datasets, all_eval_splits):

    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )
    print(max_label_length)
    audio_column_name = data_args.audio_column_name
    num_workers = 1
    dataloader_num_workers = 1
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]

    print('initializing prepare_vectorized_dataset')
    # if data_args.max_samples_per_split is not None:
    #     for split in all_eval_splits:
    #         raw_datasets[split] = (
    #             raw_datasets[split].take(data_args.max_samples_per_split)
    #             if data_args.streaming
    #             else raw_datasets[split].select(range(data_args.max_samples_per_split))
    #         )

    def prepare_eval_dataset(batch):
        # process audio input
        sample = batch["audio"]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_features"] = inputs.input_features[0]
        batch["input_length"] = len(sample["array"])

        # process targets - for evaluation these are the ground-truth transcriptions
        input_str = batch["text"]
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())

    print('preprocessing dataset')
    print(raw_datasets_features)

    vectorized_datasets = DatasetDict()

    for eval_split in all_eval_splits:
        vectorized_datasets[eval_split] = raw_datasets[eval_split].map(
            prepare_eval_dataset,
            remove_columns=raw_datasets_features,
            num_proc=num_workers,
            desc="preprocess dataset",
        )
        
    return vectorized_datasets, max_label_length

def prepare_normilazer(language, tokenizer=None):

    if language is not None:
        normalizer = BasicTextNormalizer()
    else:
        normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    return normalizer