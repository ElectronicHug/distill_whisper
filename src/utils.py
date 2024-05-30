from torch import float16, bfloat16, float32
import logging

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger

def prepare_accelerator(input_dtype, project_name, training_args):
    if input_dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = float16
    elif input_dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = float32

    kwargs = InitProcessGroupKwargs()

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=[kwargs],
    )

    accelerator.init_trackers(project_name=project_name)

    logger = get_logger(__name__)

    # 3. Set-up basic logging
    # Create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    return accelerator, torch_dtype, logger