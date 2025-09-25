import time
import os
import pickle
import asyncio
import socket
from typing import Iterable, List, Optional, Type, Union

from transformers import GenerationConfig, PreTrainedTokenizer

import vllm
from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, LoadConfig,
                         LoRAConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig,
                         VisionLanguageConfig)
from vllm.core.scheduler import (ScheduledSequenceGroup, Scheduler,
                                 SchedulerOutputs)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (ExecuteModelRequest, MultiModalData, SamplerOutput,
                           Sequence, SequenceGroup, SequenceGroupMetadata,
                           SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer_group import (BaseTokenizerGroup,
                                                     get_tokenizer_group)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import Counter, socket_recv, is_cpu

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


def _load_generation_config_dict(model_config: ModelConfig):
    try:
        return GenerationConfig.from_pretrained(
            model_config.model,
            revision=model_config.revision,
            do_sample=True
        ).to_diff_dict()
    except OSError:
        # Not found.
        return {}


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        device_config: The configuration related to the device.
        lora_config (Optional): The configuration related to serving multi-LoRA.
        vision_language_config (Optional): The configuration related to vision
            language models.
        speculative_config (Optional): The configuration related to speculative
            decoding.
        executor_class: The model executor class for managing distributed
            execution.
        log_stats: Whether to log statistics.
        usage_context: Specified entry point, used for usage info collection
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        model_executor: Optional[GPUExecutorAsync] = None,
        ip_list: Optional[List[str]] = None
    ) -> None:
        engine_start_time = time.time()
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "tokenizer_revision=%s, trust_remote_code=%s, dtype=%s, "
            "max_seq_len=%d, download_dir=%r, load_format=%s, "
            "tensor_parallel_size=%d, disable_custom_all_reduce=%s, "
            "quantization=%s, enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, seed=%d, served_model_name=%s)",
            vllm.__version__,
            model_config.model,
            speculative_config,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            model_config.tokenizer_mode,
            model_config.revision,
            model_config.tokenizer_revision,
            model_config.trust_remote_code,
            model_config.dtype,
            model_config.max_model_len,
            load_config.download_dir,
            load_config.load_format,
            parallel_config.tensor_parallel_size,
            parallel_config.disable_custom_all_reduce,
            model_config.quantization,
            model_config.enforce_eager,
            cache_config.cache_dtype,
            model_config.quantization_param_path,
            device_config.device,
            decoding_config,
            model_config.seed,
            model_config.served_model_name,
        )
        # TODO(woosuk): Print more configs in debug mode.
        
        max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
        print(f"Warning: limit max_model_len to {max_model_len} due to limited KV cache")
        model_config.max_model_len = max_model_len

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.vision_language_config = vision_language_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.log_stats = log_stats

        # Pipeline parallelism
        self.onfly_batch_num = 0
        self.batch_output_futures = []
        self.iteration_counter = 0
        if model_executor is not None:
            # when model_executor is not none, we should read pp-related info from model_executor because environment variables may have changed
            self.pp_rank = model_executor.pp_rank
            self.pp_size = model_executor.pp_size
            self.dest_pp_size = model_executor.dest_pp_size
        else:
            self.pp_rank = int(os.getenv('PP_RANK', '0'))
            self.pp_size = int(os.getenv('PP_SIZE', '1'))
            self.dest_pp_size = int(os.getenv('DEST_PP_SIZE', '1'))
        self.global_timer = None
        self.dest_inited = False
        self.max_batch_size = int(os.getenv('BATCH_SIZE', '16'))
        self.need_dest = False
        self.start_request_migration = False
        self.fin_request_migration = False
        self.migrate_tasks = []
        if self.dest_pp_size < self.pp_size:
            self.need_dest = True

        etime = time.time()
        print(f"Engine: process args time cost = {etime - engine_start_time} seconds")
        
        # Kubernetes: no need to create output files
        '''
        if model_executor is None:
            # create output files
            os.makedirs(self.dir_name, exist_ok=True)

            files_to_create = []
            for i in range(0, self.pp_size - 1):
                output_file_name = self.dir_name + "/output_" + str(i)
                files_to_create.append(output_file_name)
            for i in range(0, self.pp_size):
                invoke_file_name = self.dir_name + "/invoke_" + str(i)
                files_to_create.append(invoke_file_name)
            for i in range(0, self.pp_size):
                result_file_name = self.dir_name + "/result_" + str(i)
                files_to_create.append(result_file_name)
            if self.need_dest and self.dest_pp_size > 1:
                for i in range(0, self.dest_pp_size):
                    result_file_name = self.dir_name + "/dest_result_" + str(i)
                    files_to_create.append(result_file_name)
                for i in range(0, self.dest_pp_size - 1):
                    output_file_name = self.dir_name + "/dest_output_" + str(i)
                    files_to_create.append(output_file_name)
            for file_name in files_to_create:
                with open(file_name, "wb", buffering=0) as f:
                    f.write(b"")
        
            # inform stages to start
            for i in range(1, self.pp_size):
                start_file_name = self.dir_name + "/start_" + str(i)
                with open(start_file_name, "wb", buffering=0) as f:
                    f.write(b"1")
        '''

        stime = time.time()
        print(f"Engine: create output files time cost = {stime - etime} seconds")

        if not self.model_config.skip_tokenizer_init:
            self.tokenizer: BaseTokenizerGroup
            self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)
        else:
            self.detokenizer = None
            self.tokenizer = None

        self.seq_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(
            model_config)
        etime = time.time()
        print(f"Engine: initialize tokenizer and generator time cost = {etime - stime} seconds")

        if model_executor is not None:
            self.model_executor = model_executor
        else:
            self.model_executor = executor_class(
                model_config=model_config,
                cache_config=cache_config,
                parallel_config=parallel_config,
                scheduler_config=scheduler_config,
                device_config=device_config,
                lora_config=lora_config,
                vision_language_config=vision_language_config,
                speculative_config=speculative_config,
                load_config=load_config,
            )
        etime_1 = time.time()
        print(f"Engine: initialize GPU executor time cost = {etime_1 - etime} seconds")

        if model_executor is not None:
            cache_config = self.model_executor.cache_config
            self.cache_config = cache_config
        else:
            if not is_cpu():
                # init connection before initialize kv caches
                self.model_executor.init_connection(ip_list)
            self._initialize_kv_caches()
        etime_2 = time.time()
        print(f"Engine: initialize KV cache time cost = {etime_2 - etime_1} seconds")

        # If usage stat is enabled, collect relevant info.
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (
                get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype":
                    str(model_config.dtype),
                    "tensor_parallel_size":
                    parallel_config.tensor_parallel_size,
                    "block_size":
                    cache_config.block_size,
                    "gpu_memory_utilization":
                    cache_config.gpu_memory_utilization,

                    # Quantization
                    "quantization":
                    model_config.quantization,
                    "kv_cache_dtype":
                    cache_config.cache_dtype,

                    # Feature flags
                    "enable_lora":
                    bool(lora_config),
                    "enable_prefix_caching":
                    cache_config.enable_prefix_caching,
                    "enforce_eager":
                    model_config.enforce_eager,
                    "disable_custom_all_reduce":
                    parallel_config.disable_custom_all_reduce,
                })
        etime_3 = time.time()
        print(f"Engine: initialize usage stats time cost = {etime_3 - etime_2} seconds")

        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        etime_4 = time.time()
        print(f"Engine: initialize scheduler time cost = {etime_4 - etime_3} seconds")

        # Metric Logging.
        if self.log_stats:
            self.stat_logger = StatLogger(
                local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                labels=dict(model_name=model_config.served_model_name),
                max_model_len=self.model_config.max_model_len)
            self.stat_logger.info("cache_config", self.cache_config)

        etime_5 = time.time()
        print(f"Engine: initialize logger time cost = {etime_5 - etime_4} seconds")
        # Create sequence output processor, e.g. for beam search or
        # speculative decoding.
        self.output_processor = (
            SequenceGroupOutputProcessor.create_output_processor(
                self.scheduler_config,
                self.detokenizer,
                self.scheduler,
                self.seq_counter,
                self.get_tokenizer_for_seq,
                stop_checker=StopChecker(
                    self.scheduler_config.max_model_len,
                    self.get_tokenizer_for_seq,
                ),
            ))
        etime = time.time()
        
        print(f"Engine: initialize output_processor time cost = {etime - etime_5} seconds")
        print(f"Engine: total init time time cost = {etime - engine_start_time} seconds")

    def _initialize_kv_caches(self) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        num_gpu_blocks, num_cpu_blocks = (
            self.model_executor.determine_num_available_blocks())

        if self.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self.model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        stime = time.time()
        engine_config = engine_args.create_engine_config()

        # Initialize the cluster and specify the executor class.
        if engine_config.device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutor
            executor_class = NeuronExecutor
        elif engine_config.device_config.device_type == "cpu":
            from vllm.executor.cpu_executor import CPUExecutor
            executor_class = CPUExecutor
        elif engine_config.parallel_config.worker_use_ray:
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutor
            executor_class = RayGPUExecutor
        else:
            assert engine_config.parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1.")
            from vllm.executor.gpu_executor import GPUExecutor
            executor_class = GPUExecutor
        etime = time.time()
        print(f"from_engine_args: import executor time cost = {etime - stime} seconds")

        # Create the LLM engine.
        engine = cls(
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )
        etime_1 = time.time()
        print(f"from_engine_args: initialize engine time cost = {etime_1 - etime} seconds")
        return engine

    def __reduce__(self):
        # This is to ensure that the LLMEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("LLMEngine should not be pickled!")

    def __del__(self):
        # Shutdown model executor when engine is garbage collected
        # Use getattr since __init__ can fail before the field is set
        if model_executor := getattr(self, "model_executor", None):
            model_executor.shutdown()

    def get_tokenizer(self) -> "PreTrainedTokenizer":
        return self.tokenizer.get_lora_tokenizer(None)

    def get_tokenizer_for_seq(self,
                              sequence: Sequence) -> "PreTrainedTokenizer":
        return self.tokenizer.get_lora_tokenizer(sequence.lora_request)

    def _init_tokenizer(self, **tokenizer_init_kwargs):
        init_kwargs = dict(
            tokenizer_id=self.model_config.tokenizer,
            enable_lora=bool(self.lora_config),
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_input_length=None,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision)
        init_kwargs.update(tokenizer_init_kwargs)
        self.tokenizer = get_tokenizer_group(
            self.parallel_config.tokenizer_pool_config, **init_kwargs)

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config:
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)

    def _check_dest_inited(self) -> bool:
        if not self.model_executor.dest_inited:
            return False
        # extend block manager of newly allocated blocks
        stime = time.time()
        total_gpu_blocks = self.model_executor.driver_worker.cache_engine.num_gpu_blocks
        self.scheduler.block_manager.extend_gpu_blocks(total_gpu_blocks)
        etime = time.time()
        print(f"extend number of gpu blocks to {total_gpu_blocks}, time cost = {(etime - stime) * 1000.0} ms")
        return True
    
    def extend_gpu_blocks(self):
        stime = time.time()
        total_gpu_blocks = self.model_executor.driver_worker.cache_engine.num_gpu_blocks
        self.scheduler.block_manager.extend_gpu_blocks(total_gpu_blocks)
        etime = time.time()
        print(f"extend number of gpu blocks to {total_gpu_blocks}, time cost = {(etime - stime) * 1000.0} ms")
        return True

    def encode_request(
        self,
        request_id: str,  # pylint: disable=unused-argument
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(request_id=request_id,
                                                     prompt=prompt,
                                                     lora_request=lora_request)
        return prompt_token_ids

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            multi_modal_data: Multi modal data per request.

        Details:
            - Set arrival_time to the current time if it is None.
            - Set prompt_token_ids to the encoded prompt if it is None.
            - Create `best_of` number of :class:`~vllm.Sequence` objects.
            - Create a :class:`~vllm.SequenceGroup` object
              from the list of :class:`~vllm.Sequence`.
            - Add the :class:`~vllm.SequenceGroup` object to the scheduler.

        Example:
            >>> # initialize engine
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> # set request arguments
            >>> example_prompt = "Who is the president of the United States?"
            >>> sampling_params = SamplingParams(temperature=0.0)
            >>> request_id = 0
            >>>
            >>> # add the request to the engine
            >>> engine.add_request(
            >>>    str(request_id),
            >>>    example_prompt,
            >>>    SamplingParams(temperature=0.0))
            >>> # continue the request processing
            >>> ...
        """
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        max_logprobs = self.get_model_config().max_logprobs
        if (sampling_params.logprobs
                and sampling_params.logprobs > max_logprobs) or (
                    sampling_params.prompt_logprobs
                    and sampling_params.prompt_logprobs > max_logprobs):
            raise ValueError(f"Cannot request more than "
                             f"{max_logprobs} logprobs.")
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = self.encode_request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        eos_token_id = None
        if self.tokenizer:
            eos_token_id = self.tokenizer.get_lora_tokenizer(
                lora_request).eos_token_id
        else:
            logger.warning("Use None for EOS token id because tokenizer is "
                           "not initialized")
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size,
                       eos_token_id, lora_request)

        # Defensive copy of SamplingParams, which are used by the sampler,
        # this doesn't deep-copy LogitsProcessor objects
        sampling_params = sampling_params.clone()
        # Add the eos token id into the sampling_params to support min_tokens
        # processing
        if seq.eos_token_id is not None:
            sampling_params.all_stop_token_ids.add(seq.eos_token_id)
        sampling_params.update_from_generation_config(
            self.generation_config_fields)

        # Create the sequence group.
        _use_dest = self.dest_inited
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                arrival_time, lora_request, multi_modal_data, _use_dest)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.

        Details:
            - Refer to the
              :meth:`~vllm.core.scheduler.Scheduler.abort_seq_group`
              from class :class:`~vllm.core.scheduler.Scheduler`.

        Example:
            >>> # initialize engine and add a request with request_id
            >>> request_id = str(0)
            >>> # abort the request
            >>> engine.abort_request(request_id)
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_decoding_config(self) -> DecodingConfig:
        """Gets the decoding configuration."""
        return self.decoding_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()
    
    async def _get_future_ret(
        self,
        use_dest: bool = False,
        blocked: bool = False
    ) -> List[SamplerOutput]:
        if self.global_timer is None:
            self.global_timer = time.time()

        if use_dest:
            socket = self.model_executor.dest_comm_sockets[self.dest_pp_size - 1]
        else:
            socket = self.model_executor.comm_sockets[self.pp_size - 1]

        seq_outputs_bytes = socket_recv(socket, blocking=False)

        if seq_outputs_bytes is None:
            if not blocked:
                return []
            while True:
                seq_outputs_bytes = socket_recv(socket, blocking=False)
                if seq_outputs_bytes is not None:
                    break
                await asyncio.sleep(0)

        stime = time.time()

        print(f"future return: get {len(seq_outputs_bytes)} bytes")
        
        seq_outputs = pickle.loads(seq_outputs_bytes)

        etime = time.time()
        elapsed = etime - self.global_timer
        self.global_timer = etime
        
        print(f"future return: interval = {elapsed * 1000.0} ms, process time cost = {(etime - stime) * 1000.0} ms")
        return seq_outputs

    def _process_model_outputs(
        self,
        output: List[SamplerOutput],
        scheduled_seq_groups: List[ScheduledSequenceGroup],
        ignored_seq_groups: List[SequenceGroup],
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> List[RequestOutput]:
        """Apply the model output to the sequences in the scheduled seq groups.

        Returns RequestOutputs that can be returned to the client.
        """

        now = time.time()

        # Organize outputs by [sequence group][step] instead of
        # [step][sequence group].
        output_by_sequence_group = create_output_by_sequence_group(
            sampler_outputs=output, num_seq_groups=len(scheduled_seq_groups))

        # Update the scheduled sequence groups with the model outputs.
        for scheduled_seq_group, outputs, seq_group_meta in zip(
                scheduled_seq_groups, output_by_sequence_group,
                seq_group_metadata_list):
            seq_group = scheduled_seq_group.seq_group
            seq_group.in_process = False
            seq_group.update_num_computed_tokens(
                scheduled_seq_group.token_chunk_size)

            self.output_processor.process_prompt_logprob(seq_group, outputs)
            if seq_group_meta.do_sample:
                self.output_processor.process_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        for seq_group in ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        .. figure:: https://i.imgur.com/sv2HssD.png
            :alt: Overview of the step function
            :align: center

            Overview of the step function.

        Details:
            - Step 1: Schedules the sequences to be executed in the next
              iteration and the token blocks to be swapped in/out/copy.

                - Depending on the scheduling policy,
                  sequences may be `preempted/reordered`.
                - A Sequence Group (SG) refer to a group of sequences
                  that are generated from the same prompt.

            - Step 2: Calls the distributed executor to execute the model.
            - Step 3: Processes the model output. This mainly includes:

                - Decodes the relevant outputs.
                - Updates the scheduled sequence groups with model outputs
                  based on its `sampling parameters` (`use_beam_search` or not).
                - Frees the finished sequence groups.

            - Finally, it creates and returns the newly generated results.

        Example:
            >>> # Please see the example/ folder for more detailed examples.
            >>>
            >>> # initialize engine and request arguments
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> example_inputs = [(0, "What is LLM?",
            >>>    SamplingParams(temperature=0.0))]
            >>>
            >>> # Start the engine with an event loop
            >>> while True:
            >>>     if example_inputs:
            >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
            >>>         engine.add_request(str(req_id), prompt, sampling_params)
            >>>
            >>>     # continue the request processing
            >>>     request_outputs = engine.step()
            >>>     for request_output in request_outputs:
            >>>         if request_output.finished:
            >>>             # return or show the request output
            >>>
            >>>     if not (engine.has_unfinished_requests() or example_inputs):
            >>>         break
        """

        # This function is deprecated in pipeline parallel

        output = []
        # Check pipeline.
        if self.onfly_batch_num > 0:
            first_batch = self.batch_output_futures[0]
            # try to get the result of first_batch 
            res_file_name = self.dir_name + "/result_" + str(self.receive_counter)
            if os.path.exists(res_file_name):
                seq_outputs = asyncio.run(self._get_future_ret())
                if len(seq_outputs) > 0:
                    output.append((first_batch, seq_outputs))
                    self.batch_output_futures.pop(0)
                    self.onfly_batch_num -= 1

        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
            )

            if self.pp_size > 1:
                # broadcast execute_model_req to every worker
                stime = time.time()
                invoke_request = pickle.dumps(execute_model_req, protocol=-1)
                invoke_file_name = self.dir_name + "/invoke_" + str(self.invoke_counter)
                # clear file content first
                with open(invoke_file_name, "wb", buffering=0) as f:
                    f.write(invoke_request)
                self.invoke_counter = (self.invoke_counter + 1) % self.pp_size
                etime = time.time()
                print(f"req len = {len(invoke_request)} bytes, invoke time cost = {(etime - stime) * 1000.0} ms")

            token_output = self.model_executor.execute_model(
                execute_model_req=execute_model_req)
            if self.pp_size == 1:
                output = token_output
            else:
                # create a future that waits its execution
                seq_ids = [seq_group_metadata.request_id for seq_group_metadata in seq_group_metadata_list]
                self.batch_output_futures.append((scheduler_outputs.scheduled_seq_groups, seq_group_metadata_list))
                self.onfly_batch_num += 1

                # mark all request to be in process
                for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
                    scheduled_seq_group.seq_group.in_process = True

                if self.onfly_batch_num == self.pp_size:
                    seq_outputs = asyncio.run(self._get_future_ret(blocked=True))
                    first_batch = self.batch_output_futures[0]
                    output.append((first_batch, seq_outputs))
                    self.batch_output_futures.pop(0)
                    self.onfly_batch_num -= 1
        
        if len(output) == 0:
            return []
        
        assert len(output) == 1

        if self.pp_size > 1:
            # format data in futures
            scheduler_outputs.scheduled_seq_groups = output[0][0][0]
            seq_group_metadata_list = output[0][0][1]
            output = output[0][1]
            
        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)

        # Log stats.
        self.do_log_stats(scheduler_outputs, output)

        return request_outputs

    def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        """Forced log when no requests active."""
        if self.log_stats:
            self.stat_logger.log(
                self._get_stats(scheduler_outputs, model_output))

    def _get_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs],
            model_output: Optional[List[SamplerOutput]] = None) -> Stats:
        """Get Stats to be Logged to Prometheus.

        Args:
            scheduler_outputs: Optional, used to populate metrics related to
                the scheduled batch,
            model_output: Optional, used to emit speculative decoding metrics
                which are created by the workers.
        """
        now = time.time()

        # System State
        #   Scheduler State
        num_running_sys = len(self.scheduler.running)
        num_swapped_sys = len(self.scheduler.swapped)
        num_waiting_sys = len(self.scheduler.waiting)

        # KV Cache Usage in %
        num_total_gpu = self.cache_config.num_gpu_blocks
        num_free_gpu = self.scheduler.block_manager.get_num_free_gpu_blocks()
        gpu_cache_usage_sys = 1.0 - (num_free_gpu / num_total_gpu)

        num_total_cpu = self.cache_config.num_cpu_blocks
        cpu_cache_usage_sys = 0.
        if num_total_cpu > 0:
            num_free_cpu = self.scheduler.block_manager.get_num_free_cpu_blocks(
            )
            cpu_cache_usage_sys = 1.0 - (num_free_cpu / num_total_cpu)

        # Iteration stats
        num_prompt_tokens_iter = 0
        num_generation_tokens_iter = 0
        time_to_first_tokens_iter: List[float] = []
        time_per_output_tokens_iter: List[float] = []

        # Request stats
        #   Latency
        time_e2e_requests: List[float] = []
        #   Metadata
        num_prompt_tokens_requests: List[int] = []
        num_generation_tokens_requests: List[int] = []
        best_of_requests: List[int] = []
        n_requests: List[int] = []
        finished_reason_requests: List[str] = []

        # NOTE: This loop assumes prefill seq_groups are before
        # decode seq_groups in scheduled_seq_groups.
        if scheduler_outputs is not None:
            num_generation_tokens_from_prefill_groups = 0.
            # NOTE: if scheduler_outputs.num_prefill_groups > 0 and
            # the len of scheduler_outputs.scheduled_seq_groups is !=
            # scheduler_outputs.num_prefill_groups, this means that
            # chunked prefills have been detected.

            for idx, scheduled_seq_group in enumerate(
                    scheduler_outputs.scheduled_seq_groups):
                group_was_prefill = idx < scheduler_outputs.num_prefill_groups
                seq_group = scheduled_seq_group.seq_group

                # NOTE: a seq_group that completed all of its prefill tokens
                # in the last iteration will have seq_group.is_prefill() = False
                # with group_was_prefill = True
                if group_was_prefill:
                    # Number of prompt tokens.
                    num_prompt_tokens_iter += (
                        scheduled_seq_group.token_chunk_size)

                    # If the seq_group just finished the prefill state
                    # get TTFT.
                    if not seq_group.is_prefill():
                        latency = seq_group.get_last_latency(now)
                        time_to_first_tokens_iter.append(latency)

                        # One generation token per finished prefill.
                        num_generation_tokens_from_prefill_groups += (
                            seq_group.num_seqs())
                else:
                    # TPOTs.
                    latency = seq_group.get_last_latency(now)
                    time_per_output_tokens_iter.append(latency)

                # Because of chunked prefill, we can have a single sequence
                # group that does multiple prompt_runs. To prevent logging
                # the same metadata more than once per request, we standardize
                # on logging request level information for finished requests,
                # which can only happen once.
                if seq_group.is_finished():
                    # Latency timings
                    time_e2e_requests.append(now -
                                             seq_group.metrics.arrival_time)

                    # Metadata
                    num_prompt_tokens_requests.append(
                        len(seq_group.prompt_token_ids))
                    num_generation_tokens_requests.extend([
                        seq.get_output_len()
                        for seq in seq_group.get_finished_seqs()
                    ])
                    best_of_requests.append(seq_group.sampling_params.best_of)
                    n_requests.append(seq_group.sampling_params.n)
                    finished_reason_requests.extend([
                        SequenceStatus.get_finished_reason(seq.status)
                        for seq in seq_group.get_finished_seqs()
                    ])

            # Number of generation tokens.
            #   num_batched_tokens equals the number of prompt_tokens plus the
            #   number of decode_tokens in a single iteration. So,
            #   num_generation_tokens = num_batched_tokens - num_prompt_tokens
            #   + num_generation_tokens_from_prefill_groups (since we generate
            #   one token on prefills on iters where the prefill finishes).
            num_generation_tokens_iter = (
                scheduler_outputs.num_batched_tokens - num_prompt_tokens_iter +
                num_generation_tokens_from_prefill_groups)

        # Spec decode, if enabled, emits specialized metrics from the worker in
        # sampler output.
        if model_output and (model_output[0].spec_decode_worker_metrics
                             is not None):
            spec_decode_metrics = model_output[0].spec_decode_worker_metrics
        else:
            spec_decode_metrics = None

        return Stats(
            now=now,

            # System stats
            #   Scheduler State
            num_running_sys=num_running_sys,
            num_swapped_sys=num_swapped_sys,
            num_waiting_sys=num_waiting_sys,
            #   KV Cache Usage in %
            gpu_cache_usage_sys=gpu_cache_usage_sys,
            cpu_cache_usage_sys=cpu_cache_usage_sys,

            # Iteration stats
            num_prompt_tokens_iter=num_prompt_tokens_iter,
            num_generation_tokens_iter=num_generation_tokens_iter,
            time_to_first_tokens_iter=time_to_first_tokens_iter,
            time_per_output_tokens_iter=time_per_output_tokens_iter,
            spec_decode_metrics=spec_decode_metrics,

            # Request stats
            #   Latency
            time_e2e_requests=time_e2e_requests,
            #   Metadata
            num_prompt_tokens_requests=num_prompt_tokens_requests,
            num_generation_tokens_requests=num_generation_tokens_requests,
            best_of_requests=best_of_requests,
            n_requests=n_requests,
            finished_reason_requests=finished_reason_requests,
        )

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> List[int]:
        return self.model_executor.list_loras()

    def check_health(self) -> None:
        self.model_executor.check_health()
