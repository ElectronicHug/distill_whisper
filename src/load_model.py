from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    get_scheduler,
    set_seed,
)

def load_config_feature_ext_tokenizer(model_name_or_path, model_args):
    # 7. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    # override timestamp tokens until tokenizer issues are fixed in transformers
    timestamps = [AddedToken("<|%.2f|>" % (i * 0.02), lstrip=False, rstrip=False) for i in range(1500 + 1)]
    tokenizer.add_tokens(timestamps)

    return config, feature_extractor, tokenizer

def load_processor(processor_path, model_args):
    processor = WhisperProcessor.from_pretrained(
        processor_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    return processor

def load_whisper_model(model_path, model_args, dtype):
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    return model


