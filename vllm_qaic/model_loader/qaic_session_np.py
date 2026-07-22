# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import json
import os
import sys
from queue import Queue
from typing import Any

import numpy as np

from vllm_qaic.logger import init_logger

try:
    import qaicrt
except ImportError:
    import platform
    import sys

    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    import qaicrt
try:
    import QAicApi_pb2 as aicapi
except ImportError:
    import sys

    sys.path.append("/opt/qti-aic/dev/python")
import QAicApi_pb2 as aicapi

logger = init_logger(__name__)

aic_to_np_dtype_mapping = {
    aicapi.FLOAT_TYPE: np.dtype(np.float32),
    aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
    aicapi.INT8_Q_TYPE: np.dtype(np.int8),
    aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
    aicapi.INT16_Q_TYPE: np.dtype(np.int16),
    aicapi.INT32_Q_TYPE: np.dtype(np.int32),
    aicapi.INT32_I_TYPE: np.dtype(np.int32),
    aicapi.INT64_I_TYPE: np.dtype(np.int64),
    aicapi.INT8_TYPE: np.dtype(np.int8),
}
VLLM_QAIC_PREFILL_QUEUE_LEN_ENV = "VLLM_QAIC_PREFILL_QUEUE_LEN"
VLLM_QAIC_ASYNC_SCHEDULING_EXEC_TIMEOUT_ENV = "VLLM_QAIC_ASYNC_SCHEDULING_EXEC_TIMEOUT"
VLLM_KV_CACHE_PREFIX = "vllmKvCache"
VLLM_QAIC_USE_LEGACY_SLICING_SPEC_ENV = "VLLM_QAIC_USE_LEGACY_SLICING_SPEC"
VLLM_QAIC_ENABLE_LRT_DEBUG_ENV = "VLLM_QAIC_ENABLE_LRT_DEBUG"


class QAICInferenceSession:
    def __init__(
        self,
        qpc_path: str,
        full_batch_size: int,
        device_ids: list[int] | None = None,
        activate: bool = True,
        stages: int | None = 1,
        cluster_id: str | None = None,
        use_async_scheduling: bool = False,
    ):
        self.stages: int = stages if stages is not None else 1
        self.cluster_id = cluster_id
        self.full_batch_size = full_batch_size

        self.decode_execObj_idx: int | None
        if cluster_id == "decode":
            self.prefill_num_execObj = 0
            self.decode_num_execObj = 1
            self.decode_execObj_idx = 0
        elif cluster_id == "prefill":
            self.prefill_num_execObj = int(
                os.getenv(VLLM_QAIC_PREFILL_QUEUE_LEN_ENV, self.stages + 1)
            )
            self.decode_num_execObj = 0
            self.decode_execObj_idx = None
        else:
            # one additional prefill_exeObj in case of async scheduling
            # so that we can early enqueue to LRT
            prefill_num_execObj = 2 if use_async_scheduling else 1
            self.prefill_num_execObj = int(
                os.getenv(VLLM_QAIC_PREFILL_QUEUE_LEN_ENV, prefill_num_execObj)
            )
            self.decode_num_execObj = 1
            self.decode_execObj_idx = 0
        self.async_scheduling_exec_timeout = int(
            os.getenv(VLLM_QAIC_ASYNC_SCHEDULING_EXEC_TIMEOUT_ENV, 300)
        )

        logger.debug(
            "Async scheduling enabled: %s, number of prefill exec objs: %s,"
            " number of decode exec objs: %s",
            use_async_scheduling,
            self.prefill_num_execObj,
            self.decode_num_execObj,
        )
        self.queue_len = self.prefill_num_execObj + self.decode_num_execObj
        self.execObj = [qaicrt.ExecObj] * (self.queue_len)

        self.prefill_available_exec_objs: Queue[int] = Queue()

        prefill_start = self.decode_num_execObj
        for i in range(prefill_start, prefill_start + self.prefill_num_execObj):
            self.prefill_available_exec_objs.put(i)

        # Load QPC
        if device_ids is not None:
            devices = qaicrt.QIDList(device_ids)
            self.context = qaicrt.Context(devices)
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)  # Async API

        # Use Environment variable to enable LRT Debug Logs
        enable_debug_logs = os.getenv(VLLM_QAIC_ENABLE_LRT_DEBUG_ENV, "0") == "1"
        if enable_debug_logs:
            assert (
                self.context.setLogLevel(qaicrt.QLogLevel.QL_DEBUG)
                == qaicrt.QStatus.QS_SUCCESS
            ), "Failed to setLogLevel"
        qpc = qaicrt.Qpc(str(qpc_path))
        # Load IO Descriptor
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to getIoDescriptor"
        iodesc.ParseFromString(bytes(iodesc_data))
        self.allowed_shapes = [
            [
                (aic_to_np_dtype_mapping[x.type].itemsize, list(x.dims))
                for x in allowed_shape.shapes
            ]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map = {
            binding.name: binding.index for binding in self.bindings
        }

        # Create and load Program
        prog_properties = qaicrt.QAicProgramProperties()
        prog_properties.dataPathTimeoutMs = 60_000
        queueProperties = qaicrt.QAicQueueProperties()
        queueProperties.numThreadsPerQueue = 1
        self.queue.initProperties(queueProperties)

        dev_id_non_mq = None
        if device_ids:
            if len(device_ids) == 1:
                dev_id_non_mq = device_ids[0]
            elif len(device_ids) > 1:
                prog_properties.devMapping = ":".join(map(str, device_ids))
        self.program = qaicrt.Program(self.context, dev_id_non_mq, qpc, prog_properties)
        assert self.program.load() == qaicrt.QStatus.QS_SUCCESS, (
            "Failed to load program"
        )
        self.activate_done = False
        if activate:
            self.activate()
        # Create input qbuffers and buf_dims
        self.qbuffers = [
            [qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings]
            for _ in range(self.queue_len)
        ]
        self.buf_dims = [
            qaicrt.BufferDimensionsVecRef(
                [
                    (
                        aic_to_np_dtype_mapping[binding.type].itemsize,
                        list(binding.dims),
                    )
                    for binding in self.bindings
                ]
            )
            for _ in range(self.queue_len)
        ]

        self.use_legacy_slicing_spec = (
            os.getenv(VLLM_QAIC_USE_LEGACY_SLICING_SPEC_ENV, "0") == "1"
        )
        self.cluster_id = cluster_id

        self.decode_buff_map = []
        for name in self.input_names:
            if self._is_kv_cache_name(name):
                self.decode_buff_map.append((name, self.binding_index_map[name]))

        # Sort by layer number, then alphabetically within each layer.
        # Standard attn: past_key < past_value. MLA: compressed_kv < k_pe.
        def _kv_sort_key(item):
            name = item[0]
            part = name.split(".")[1] if "." in name else "0"
            return int(part.split("_")[0]), name

        self.decode_buff_map.sort(key=_kv_sort_key)

        self.prefill_buff_map = []
        for name in self.output_names:
            if self._is_kv_cache_name(name) and name.endswith("RetainedState"):
                self.prefill_buff_map.append(
                    (
                        name.replace("_RetainedState", ""),
                        self.binding_index_map[name],
                    )
                )
        # sort by layer number grouping keys and values
        self.prefill_buff_map.sort(key=_kv_sort_key)

        self.kv_cache_info = []
        for name, _ in self.decode_buff_map:
            if self._is_kv_cache_name(name):
                _binding = self.bindings[self.binding_index_map[name]]
                kv_shape = tuple(_binding.dims)
                kv_type = aic_to_np_dtype_mapping[_binding.type]
                kv_size = _binding.size
                self.kv_cache_info.append((kv_shape, kv_type, kv_size))

        # Hybrid KV detected if more than one KV shape found
        _num_kv_cache_info = len(set(self.kv_cache_info))
        self.is_hybrid_kv = _num_kv_cache_info > 1

        self.kv_slicing_spec_handle = None
        if _num_kv_cache_info:
            buffer_spec_json = (
                self.get_json_for_kv_cache_slicing()
                if not self.is_hybrid_kv
                else self.get_json_for_full_kv_cache_slicing()
            )
            self.kv_slicing_spec_handle = self.get_slicing_spec_handle(
                buffer_spec_json=buffer_spec_json
            )
        self.repetition_penalty_spec_handle = None
        past_repetition_penalty_buffer = "past_repetition_penalty_buffer"
        if past_repetition_penalty_buffer in self.input_names:
            self.repetition_penalty_map = [
                (
                    past_repetition_penalty_buffer,
                    self.binding_index_map[past_repetition_penalty_buffer],
                )
            ]
            buffer_spec_json = self.get_json_for_repetition_penalty_slicing()
            self.repetition_penalty_spec_handle = self.get_slicing_spec_handle(
                buffer_spec_json=buffer_spec_json
            )

        for name in self.output_names:
            if name.startswith("log"):
                self.prefill_buff_map.append((name, self.binding_index_map[name]))

        for y in range(self.queue_len):
            self.skip_buffers(
                [x for x in self.input_names if self._is_kv_cache_name(x)],
                y,
            )
            self.skip_buffers(
                [x for x in self.output_names if x.endswith("_RetainedState")],
                y,
            )

    def _is_kv_cache_name(self, name: str) -> bool:
        if self.use_legacy_slicing_spec:
            return name.startswith("past_key") or name.startswith("past_value")
        return VLLM_KV_CACHE_PREFIX in name

    def get_json_for_full_kv_cache_slicing(self):
        buffer_specs = []
        _full_attn_dim_spec = [
            {"start": "batch_index"},
            {"start": 0},
            {"start": "ctx_start"},
            {"start": 0},
        ]
        _linear_attn_dim_spec = [
            {"start": "batch_index"},
            {"start": 0},
            {"start": 0},
        ]
        for binding in self.bindings:
            name = binding.name
            ndim = len(binding.dims)
            if self._is_kv_cache_name(name) and name.endswith("_RetainedState"):
                size = aic_to_np_dtype_mapping[binding.type]
                base_name = name.replace("_RetainedState", "")
                buffer_specs.append(
                    {
                        "Name": f"{base_name}.*",
                        "ElemSize": size.itemsize,
                        "DimSpecs": _full_attn_dim_spec
                        if ndim == 4
                        else _linear_attn_dim_spec,
                    }
                )
        json_spec = {"BufferSpecs": buffer_specs}
        return json.dumps(json_spec)

    def get_json_for_kv_cache_slicing(self):
        # All KVs have same shape/size/type
        assert not self.is_hybrid_kv, (
            "Uniform json spec incompatible with hybrid KV cache"
        )
        size = self.kv_cache_info[0][1]

        if self.use_legacy_slicing_spec:
            json_spec = {
                "BufferSpecs": [
                    {
                        "Name": "past_key.*",
                        "ElemSize": size.itemsize,
                        "DimSpecs": [
                            {"start": "batch_index"},
                            {"start": 0},
                            {"start": "ctx_start"},
                            {"start": 0},
                        ],
                    },
                    {
                        "Name": "past_value.*",
                        "ElemSize": size.itemsize,
                        "DimSpecs": [
                            {"start": "batch_index"},
                            {"start": 0},
                            {"start": "ctx_start"},
                            {"start": 0},
                        ],
                    },
                ]
            }
        else:
            json_spec = {
                "BufferSpecs": [
                    {
                        "Name": f".*{VLLM_KV_CACHE_PREFIX}.*",
                        "ElemSize": size.itemsize,
                        "DimSpecs": [
                            {"start": "batch_index"},
                            {"start": 0},
                            {"start": "ctx_start"},
                            {"start": 0},
                        ],
                    }
                ]
            }
        return json.dumps(json_spec)

    def get_json_for_repetition_penalty_slicing(self):
        size = aic_to_np_dtype_mapping[
            self.bindings[self.binding_index_map["past_repetition_penalty_buffer"]].type
        ]
        json_spec = {
            "BufferSpecs": [
                {
                    "Name": "past_repetition_penalty_buffer",
                    "ElemSize": size.itemsize,
                    "DimSpecs": [
                        {"start": "batch_index"},
                        {"start": 0},
                    ],
                }
            ]
        }
        return json.dumps(json_spec)

    def get_slicing_spec_handle(self, buffer_spec_json):
        status, slicingSpecHandle = self.program.createSlicingSpecHandle(
            buffer_spec_json
        )
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to create SlicingSpecHandle"
        return slicingSpecHandle

    @property
    def input_names(self) -> list[str]:
        return [
            binding.name
            for binding in self.bindings
            if binding.dir == aicapi.BUFFER_IO_TYPE_INPUT
        ]

    @property
    def output_names(self) -> list[str]:
        return [
            binding.name
            for binding in self.bindings
            if binding.dir == aicapi.BUFFER_IO_TYPE_OUTPUT
        ]

    def get_bindings(self, binding_names: list[str]) -> list[aicapi.IoBinding]:
        bindings: list[aicapi.IoBinding] = [
            binding for binding in self.bindings if binding.name in binding_names
        ]
        return bindings

    def get_bindings_shapes(
        self, binding_name: list[str]
    ) -> dict[str, list[list[int]]]:
        """Function returns all possible shapes of requested buffers

        Args:
            binding_name (List[str]): List of I/O buffer names

        Returns:
            Dict[str, List[List[int]]]: All possible shapes of requested buffers
        """
        result: dict[str, list[list[int]]] = {}
        for name in binding_name:
            try:
                result[name] = []
                idx = self.binding_index_map[name]
            except Exception:
                logger.warning("Unable to find binding: %s", name)
                continue
            for allowed_shaped in self.allowed_shapes:
                result[name].append(allowed_shaped[idx][1])
        return result

    def get_logits_ndim(self) -> int:
        """Returns the number of dimensions of the logits output binding.
        Defaults to 3 if the binding is absent or shapes are unavailable.
        """
        try:
            shapes = self.get_bindings_shapes(["logits"])
            logits_shapes = shapes.get("logits", [])
            if not logits_shapes:
                return 3
            return len(logits_shapes[0])
        except Exception:
            logger.warning("Unable to determine logits ndim, defaulting to 3")
            return 3

    def activate(self):
        self.activate_done = True
        self.program.activate()
        for i in range(self.queue_len):
            self.execObj[i] = qaicrt.ExecObj(self.context, self.program)

    def deactivate(self):
        print("Deactivating qpc..")
        if self.activate_done:
            self.program.deactivate()
            self.activate_done = False

    def set_buffers(self, buffers: dict[str, np.ndarray], index: int = 0):
        for buffer_name, buffer in buffers.items():
            if buffer_name not in self.binding_index_map:
                logger.warning("Buffer: %s not found", buffer_name)
                continue
            buffer_index: int = self.binding_index_map[buffer_name]
            contiguous = np.ascontiguousarray(buffer)
            if contiguous is not buffer:
                logger.warning(
                    "Non-contingous buffer used while set_buffers."
                    " Copying data to a continguous buffer."
                )
                buffers[buffer_name] = contiguous
            buffer = contiguous
            self.qbuffers[index][buffer_index] = qaicrt.QBuffer(buffer)
            self.buf_dims[index][buffer_index] = (
                buffer.itemsize,
                buffer.shape if len(buffer.shape) > 0 else (1,),
            )

    def unskip_buffers(self, skipped_buffer_names: list[str], index: int = 0) -> None:
        if not skipped_buffer_names:
            return
        bindings: list[aicapi.IoBinding] = self.get_bindings(skipped_buffer_names)
        buffers: dict[str, np.ndarray] = dict()
        for binding in bindings:
            aic_dtype: int = binding.type
            np_dtype: np.dtype = aic_to_np_dtype_mapping[aic_dtype]
            dims: list[int] = binding.dims
            arr = np.zeros(dims, dtype=np_dtype)
            buffers[binding.name] = arr
        self.set_buffers(buffers, index)

    def skip_buffers(self, skipped_buffer_names: list[str], index: int = 0):
        self.set_buffers({k: np.array([]) for k in skipped_buffer_names}, index)

    def get_tuple_list_from_dict(self, dict_in):
        # Convert the buffer_dict to a list of tuples
        buffer_idx_to_buffer = []
        for buffer_name, buffer in dict_in.items():
            if buffer_name not in self.binding_index_map:
                logger.warning("Buffer: %s not found", buffer_name)
                continue
            buffer_index: int = self.binding_index_map[buffer_name]
            if buffer is None:
                continue
            buffer_idx_to_buffer.append((buffer_index, buffer))
        return buffer_idx_to_buffer

    def extract_outputs(self, input_dict):
        output_dict = dict()
        for bufname in self.output_names:
            if bufname in input_dict:
                output_dict[bufname] = input_dict[bufname]
        return output_dict

    def create_numpy_buffers(self, input_dict, direction, shape, size):
        bufnames = []
        if direction == "in":
            bufnames = [n for n in self.input_names if self._is_kv_cache_name(n)]
        elif direction == "out":
            bufnames = [
                n
                for n in self.output_names
                if self._is_kv_cache_name(n) and n.endswith("_RetainedState")
            ]
        else:
            raise ValueError("invalid buffer direction to create_numpy_buffers")
        for bufname in bufnames:
            if len(shape) == 0:
                input_dict[bufname] = np.array([])
            else:
                input_dict[bufname] = np.zeros(shape=shape, dtype=size)

    def create_numpy_penalty_buffers(self, input_dict, direction, shape, dtype):
        bufnames = []
        if direction == "in":
            bufnames = [
                "past_repetition_penalty_buffer",
                "past_presence_penalty_buffer",
            ]
        elif direction == "out":
            bufnames = [
                "past_repetition_penalty_buffer_RetainedState",
                "past_presence_penalty_buffer_RetainedState",
            ]
        else:
            raise ValueError("invalid buffer direction to create_numpy_penalty_buffers")
        for bufname in bufnames:
            if len(shape) == 0:
                input_dict[bufname] = np.array([])
            else:
                input_dict[bufname] = np.zeros(shape=shape, dtype=dtype)

    def create_output_buffers(self, input_dict, shape, size, buffer_name="logits"):
        if buffer_name not in self.binding_index_map:
            logger.warning("Buffer: %s not found", buffer_name)
            return
        input_dict[buffer_name] = np.empty(shape=shape, dtype=size)

    def set_data_for_kv_handoff(
        self,
        kv_cache_buffers,
        slicing_parameters,
        index=0,
        buff_map: list[tuple[str, int]] | None = None,
    ):
        return self._set_data_with_slices(
            kv_cache_buffers,
            slicing_parameters,
            self.kv_slicing_spec_handle,
            index,
            buff_map,
        )

    def set_data_for_repetition_penalty(
        self, repetition_penalty_buffers, slicing_parameters, index=0
    ):
        return self._set_data_with_slices(
            repetition_penalty_buffers,
            slicing_parameters,
            self.repetition_penalty_spec_handle,
            index,
            self.repetition_penalty_map,
        )

    def _set_data_with_slices(
        self,
        buffers,
        slicing_parameters,
        slicing_spec_handle,
        index=0,
        buff_map: list[tuple[str, int]] | None = None,
    ):
        if isinstance(buffers, (list, np.ndarray)):
            assert buff_map is not None
            assert len(buffers) == len(buff_map) or len(buffers) + 1 == len(buff_map), (
                "buffers must be a list of numpy arrays or a dictionary of numpy arrays"
            )
            slices_as_tuple_list = [
                (name[1], buff) for name, buff in zip(buff_map, buffers, strict=False)
            ]
        else:
            slices_as_tuple_list = self.get_tuple_list_from_dict(buffers)
        status, slicingHandle = self.execObj[index].setDataWithSlices(
            slices_as_tuple_list, slicing_spec_handle, slicing_parameters
        )
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to setDataWithSlices"
        return buffers

    def _make_inputs_contiguous(self, inputs: dict) -> None:
        for k, v in inputs.items():
            inputs[k] = np.ascontiguousarray(v)

    def np_run(
        self,
        inputs: dict[str, Any],
        slicing_parameters: list[tuple[str, int]] | None = None,
        is_prefill=True,
    ) -> int:
        # Will block here if no more exec objects are ready
        logger.debug(
            "Waiting to allocate exec obj for %s", "Prefill" if is_prefill else "Decode"
        )
        if is_prefill:
            exec_obj_idx = self.prefill_available_exec_objs.get(
                timeout=self.async_scheduling_exec_timeout
            )
        else:
            assert self.decode_execObj_idx is not None
            exec_obj_idx = self.decode_execObj_idx

        self._make_inputs_contiguous(inputs)
        # setdata with slices for each instance of the sliceddata
        slices_as_tuple_list = self.get_tuple_list_from_dict(inputs)
        if slicing_parameters is None:
            status = self.execObj[exec_obj_idx].setData(slices_as_tuple_list)
            assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to setDataWithSlices"
        else:
            status, slicingHandle = self.execObj[exec_obj_idx].setDataWithSlices(
                slices_as_tuple_list, self.kv_slicing_spec_handle, slicing_parameters
            )
            assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to setDataWithSlices"
        try:
            assert (
                self.queue.enqueue(self.execObj[exec_obj_idx])
                == qaicrt.QStatus.QS_SUCCESS
            ), "Failed to enqueue"
        except Exception as e:
            logger.error("Error while enqueuing %s", e)
            return 0
        return exec_obj_idx

    def np_run_pipeline(
        self,
        inputs: dict[str, np.ndarray],
        slicing_parameters: list[tuple[str, int]] | None = None,
        last_chunk: bool = False,
        kv_cache_buffers=None,
    ) -> int:
        logger.debug("Waiting to allocate exec_obj for pipeline prefill")
        exec_obj_idx = self.prefill_available_exec_objs.get(
            timeout=self.async_scheduling_exec_timeout
        )
        logger.debug("Got an exec_obj %s", exec_obj_idx)

        if last_chunk:
            assert kv_cache_buffers is not None, (
                "Found None KV buffer to load KV caches"
            )
            batch_index = int(inputs["batch_index"].item())
            self.set_data_for_kv_handoff(
                kv_cache_buffers,
                [("batch_index", batch_index % self.full_batch_size), ("ctx_start", 0)],
                exec_obj_idx,
                self.prefill_buff_map[:-1],
            )

        self._make_inputs_contiguous(inputs)
        slices_as_tuple_list = self.get_tuple_list_from_dict(inputs)

        if slicing_parameters is None:
            status = self.execObj[exec_obj_idx].setData(slices_as_tuple_list)
            assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to setData"
        else:
            slicing_spec_handle = self.kv_slicing_spec_handle
            status, slicingHandle = self.execObj[exec_obj_idx].setDataWithSlices(
                slices_as_tuple_list, slicing_spec_handle, slicing_parameters
            )
            assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to setDataWithSlices"
        assert (
            self.queue.enqueue(self.execObj[exec_obj_idx]) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to enqueue"

        return exec_obj_idx

    def complete_inf(self, index: int, is_prefill: bool):
        if self.execObj[index].waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"
            # Print additional error messages for unmatched dimension error
            if self.allowed_shapes:
                error_message += "\n\n"
                error_message += (
                    '(Only if "No matching dimension found" error is present above)'
                )
                error_message += "\nAllowed shapes:"
                for i, allowed_shape in enumerate(self.allowed_shapes):
                    error_message += f"\n{i}\n"
                    for binding, (elemsize, shape), (_, passed_shape) in zip(
                        self.bindings,
                        allowed_shape,
                        self.buf_dims[index],
                        strict=False,
                    ):
                        if passed_shape[0] == 0:
                            if not binding.is_partial_buf_allowed:
                                logger.warning(
                                    "Partial buffer not allowed for: %s", {binding.name}
                                )
                            continue
                        error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
                error_message += "\n\nPassed shapes:\n"
                for binding, (elemsize, shape) in zip(
                    self.bindings, self.buf_dims[index], strict=False
                ):
                    if shape[0] == 0:
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            raise ValueError(error_message)
        logger.debug("Releasing exec obj: %s", index)
        if is_prefill:
            self.prefill_available_exec_objs.put(index)
