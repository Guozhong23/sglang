"""Four CPU/Gloo ranks run production DP transport with nonzero/NaN payloads.

The SUM reference includes its existing post-gather mask. This establishes
replica de-duplication and padding semantics, not HCCL capture support.
"""

import ast
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_welm_tp2dp2_admission import SRT, load_nodes
from test_welm_tp2dp2_repairs import pack  # Install the isolated production module.


def worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        groups = [dist.new_group(members) for members in ([0, 2], [1, 3])]
        dp, lane = divmod(rank, 2)

        def reduce(value):
            dist.all_reduce(value)
            return value

        group = NS(all_reduce=reduce)
        replica = NS(
            world_size=2,
            rank_in_group=dp,
            all_gather_into_tensor=lambda out, value: dist.all_gather_into_tensor(
                out, value, group=groups[lane]
            ),
        )
        gather = load_nodes(
            SRT / "layers/dp_attention.py",
            ["welm_dp_replicate_gather"],
            {"torch": torch},
        )["welm_dp_replicate_gather"]
        for slots in ((4, 4), (4, 2), (0, 4), (0, 0), (4, 4)):
            for requested in ((0, 1), (1, 0), (1, 1), slots):
                counts = tuple(min(a, b) for a, b in zip(requested, slots))
                n, total = slots[dp], sum(slots)
                offsets = (0, slots[0])
                source = torch.full((n, 3), float("nan") if lane else -999.0)
                real = counts[dp]
                source[:real] = torch.arange(real * 3).reshape(real, 3) + dp * 100 + 1
                expected = torch.zeros(total, 3)
                valid = torch.zeros(total, dtype=torch.bool)
                for d in (0, 1):
                    off, rows = offsets[d], counts[d]
                    expected[off : off + rows] = (
                        torch.arange(rows * 3).reshape(rows, 3) + d * 100 + 1
                    )
                    valid[off : off + rows] = True
                view = NS(
                    slot_rows_cpu=slots,
                    slot_offsets_cpu=offsets,
                    global_slot_rows=total,
                    local_slot_rows=n,
                    real_rows_gpu=torch.tensor(counts),
                    _replicate_gather_buffer=torch.empty(total, 3),
                    _replica_gather_input=torch.empty(n, 3),
                )
                pointer = view._replicate_gather_buffer.data_ptr()
                for fused, use_replica in (
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ):
                    output = gather(
                        source,
                        view,
                        group=group,
                        dp_rank=dp,
                        contribute=lane == 0,
                        fused_pack=fused,
                        replica_dp_group=replica if use_replica else None,
                    )
                    if not fused and not use_replica:
                        output.masked_fill_(~valid[:, None], 0)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    assert output.data_ptr() == pointer
        # Real rows must match within each attention TP group. Contradicting
        # replicas cannot be made equivalent by a transport substitution.
        payload = torch.tensor([float(rank)])
        out = torch.empty(2)
        replica.all_gather_into_tensor(out, payload)
        assert out.tolist() == ([0.0, 2.0] if lane == 0 else [1.0, 3.0])
    finally:
        dist.destroy_process_group()


class TestReplicaGather(unittest.TestCase):
    def test_group_creation_defaults_and_recovery_guard(self):
        # Execute the production startup block with group creation mocked;
        # the real initialize_model_parallel default is recovered_rank=False.
        source = SRT / "distributed/parallel_state.py"
        tree = ast.parse(source.read_text())
        body = next(
            n.body
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "initialize_model_parallel"
        )
        begin = next(
            i
            for i, n in enumerate(body)
            if isinstance(n, ast.Global) and "_WELM_REPLICA_DP" in n.names
        )
        end = next(
            i
            for i in range(begin + 1, len(body))
            if isinstance(body[i], ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "moe_ep_size"
                for t in body[i].targets
            )
        )
        code = compile(
            ast.Module(body=body[begin:end], type_ignores=[]), str(source), "exec"
        )
        for recovered in (False, True):
            created = []
            fake_torch = NS(
                inference_mode=torch.inference_mode,
                distributed=NS(
                    all_gather_object=lambda out, value, **kw: out.__setitem__(
                        slice(None), [value] * 4
                    )
                ),
            )
            ns = dict(
                torch=fake_torch,
                envs=NS(SGLANG_WELMV4_DP_REPLICA_ALLGATHER=NS(get=lambda: True)),
                tensor_model_parallel_size=4,
                attention_data_parallel_size=2,
                world_size=4,
                attn_tp_size=2,
                expert_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                recovered_rank=recovered,
                rank_offset=0,
                backend="hccl",
                get_world_group=lambda: NS(
                    world_size=4, cpu_group=object(), local_rank=0
                ),
                init_model_parallel_group=lambda members, *a, **kw: created.append(
                    members
                ),
            )
            if recovered:
                with self.assertRaises(ValueError):
                    exec(code, ns)
                self.assertEqual(created, [])
            else:
                exec(code, ns)
                self.assertEqual(created, [[[0, 2], [1, 3]]])

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_four_rank_transport_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mp.spawn(
                worker, args=((Path(tmp) / "store").as_uri(),), nprocs=4, join=False
            )
            deadline = time.monotonic() + 120
            try:
                while not ctx.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("DP gather did not complete in 120 seconds")
            finally:
                for process in ctx.processes:
                    if process.is_alive():
                        process.terminate()
                for process in ctx.processes:
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
