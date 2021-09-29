r""""Contains definitions of the methods used by the _BaseDataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

from collections import namedtuple
import torch
from torch._six import queue, container_abcs, string_classes
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper
from torch.utils.data._utils.worker import _ResumeIteration


def _to_cuda_loop(in_queue, out_queue, device_id, cuda_aug_pipeline, cuda_collate_fn, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.cuda.set_device(device_id)

    iteration_end = False

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        if isinstance(r, _ResumeIteration):
            # Acknowledge the main process
            out_queue.put(r)
            iteration_end = False
            continue
        elif r is None:
            # Received the final signal
            assert done_event.is_set() or iteration_end
            break
        elif done_event.is_set() or iteration_end:
            # `done_event` is set. But I haven't received the final signal
            # (None) yet. I will keep continuing until get it, and skip the
            # processing steps.
            continue
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                if len(cuda_aug_pipeline):
                    for transform in cuda_aug_pipeline:
                        data = transform(data)
                    data = cuda_collate_fn(data)
                    torch.cuda.empty_cache()
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id))
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        del r  # save memory