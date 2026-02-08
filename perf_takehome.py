"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        # This is required to match with the yield in reference_kernel2.
        self.instrs.append({"flow": [("pause",)]})

        # Precomputing some pointer addresses from the known memory layout.
        forest_values_p_const = 7
        inp_indices_p_const = forest_values_p_const + n_nodes
        inp_values_p_const = inp_indices_p_const + batch_size

        instructions = []

        # First, load all initial accumulators into registers. Note that
        # registers all begin initialized to zero.
        zero = self.alloc_scratch("zero", 1)
        one = self.alloc_scratch("one", 1)
        three = self.alloc_scratch("three", 1)
        three_v = self.alloc_scratch("three_v", 8)
        five = self.alloc_scratch("five", 1)
        five_v = self.alloc_scratch("five_v", 8)
        eight = self.alloc_scratch("eight", 1)
        nine = self.alloc_scratch("nine", 1)
        nine_v = self.alloc_scratch("nine_v", 8)
        twelve = self.alloc_scratch("twelve", 1)
        twelve_v = self.alloc_scratch("twelve_v", 8)
        sixteen = self.alloc_scratch("sixteen", 1)
        sixteen_v = self.alloc_scratch("sixteen_v", 8)
        nineteen = self.alloc_scratch("nineteen", 1)
        nineteen_v = self.alloc_scratch("nineteen_v", 8)
        acc_lo = self.alloc_scratch("tmp1", 1)
        acc_hi = self.alloc_scratch("tmp2", 1)
        tree_node_val_zero = self.alloc_scratch("tmp3", 1)
        hash_stage0_const = self.alloc_scratch("hash_stage0_const", 1)
        hash_stage0_v = self.alloc_scratch("hash_stage0_v", 8)
        hash_stage1_const = self.alloc_scratch("hash_stage1_const", 1)
        hash_stage1_v = self.alloc_scratch("hash_stage1_v", 8)
        hash_stage2_const = self.alloc_scratch("hash_stage2_const", 1)
        hash_stage2_v = self.alloc_scratch("hash_stage2_v", 8)
        hash_stage3_const = self.alloc_scratch("hash_stage3_const", 1)
        hash_stage3_v = self.alloc_scratch("hash_stage3_v", 8)
        hash_stage4_const = self.alloc_scratch("hash_stage4_const", 1)
        hash_stage4_v = self.alloc_scratch("hash_stage4_v", 8)
        hash_stage5_const = self.alloc_scratch("hash_stage5_const", 1)
        hash_stage5_v = self.alloc_scratch("hash_stage5_v", 8)
        instructions.append(
            {
                "load": [
                    ("const", tree_node_val_zero, forest_values_p_const),
                    ("const", acc_lo, inp_values_p_const),
                ],
                "flow": [
                    ("add_imm", eight, zero, 8),
                ],
            }
        )
        instructions.append(
            {
                "load": [
                    ("load", tree_node_val_zero, tree_node_val_zero),
                    ("const", three, 3),
                ],
                "alu": [
                    ("+", sixteen, eight, eight),
                    ("+", acc_hi, eight, acc_lo),
                ],
                "flow": [
                    ("add_imm", one, zero, 1),
                ],
            }
        )

        # Start loading initial accumulator values into registers.
        for i in range(8):
            self.alloc_scratch(f"acc_{2*i}", 8)
            self.alloc_scratch(f"acc_{2*i+1}", 8)
            self.alloc_scratch(f"index_{2*i}", 8)
            self.alloc_scratch(f"index_{2*i+1}", 8)
            self.alloc_scratch(f"node_{2*i}", 8)
            self.alloc_scratch(f"node_{2*i+1}", 8)
            instruction = {
                "load": [
                    ("vload", self.scratch[f"acc_{2*i}"], acc_lo),
                    ("vload", self.scratch[f"acc_{2*i+1}"], acc_hi),
                ],
                "alu": [
                    ("+", acc_lo, acc_lo, sixteen),
                    ("+", acc_hi, acc_hi, sixteen),
                ],
                # Note that the initial indexes are all zero, and therefore
                # the initial node values are also all the same, so we can
                # vbroadcast both the indexes and node values.
                "valu": [
                    ("vbroadcast", self.scratch[f"index_{2*i}"], zero),
                    ("vbroadcast", self.scratch[f"index_{2*i+1}"], zero),
                    ("vbroadcast", self.scratch[f"node_{2*i}"], tree_node_val_zero),
                    (
                        "vbroadcast",
                        self.scratch[f"node_{2*i+1}"],
                        tree_node_val_zero,
                    ),
                ],
            }

            # This is a very hacky way of hijacking some spare slots to
            # initialize some more constants (specifically, the hashing
            # constants).
            if i == 0:
                instruction["alu"].extend(
                    [
                        (
                            "+",
                            nine,
                            one,
                            eight,
                        ),
                        (
                            "+",
                            nineteen,
                            three,
                            sixteen,
                        ),
                    ]
                )
                instruction["flow"] = [("add_imm", five, zero, 5)]
                instruction["valu"].extend(
                    [
                        ("vbroadcast", three_v, three),
                        ("vbroadcast", sixteen_v, sixteen),
                    ]
                )
            if i == 1:
                instruction["alu"].append(("+", twelve, three, nine))
                instruction["flow"] = [("add_imm", hash_stage0_const, zero, 0x7ED55D16)]
                instruction["valu"].extend(
                    [
                        ("vbroadcast", five_v, five),
                        ("vbroadcast", nine_v, nine),
                    ]
                )
            if i == 2:
                instruction["flow"] = [("add_imm", hash_stage1_const, zero, 0xC761C23C)]
                instruction["valu"].extend(
                    [
                        ("vbroadcast", twelve_v, twelve),
                        ("vbroadcast", nineteen_v, nineteen),
                    ]
                )
            if i == 3:
                instruction["flow"] = [("add_imm", hash_stage2_const, zero, 0x165667B1)]
                instruction["valu"].extend(
                    [
                        ("vbroadcast", hash_stage0_v, hash_stage0_const),
                        ("vbroadcast", hash_stage1_v, hash_stage1_const),
                    ]
                )
            if i == 4:
                instruction["flow"] = [("add_imm", hash_stage3_const, zero, 0xD3A2646C)]
                instruction["valu"].extend(
                    [
                        ("vbroadcast", hash_stage2_v, hash_stage2_const),
                    ]
                )
            if i == 5:
                instruction["flow"] = [("add_imm", hash_stage4_const, zero, 0xFD7046C5)]
                instruction["valu"].extend(
                    [
                        ("vbroadcast", hash_stage3_v, hash_stage3_const),
                    ]
                )
            if i == 6:
                instruction["flow"] = [("add_imm", hash_stage5_const, zero, 0xB55A4F09)]
                instruction["valu"].extend(
                    [
                        ("vbroadcast", hash_stage4_v, hash_stage4_const),
                    ]
                )
            if i == 7:
                instruction["valu"].extend(
                    [
                        ("vbroadcast", hash_stage5_v, hash_stage5_const),
                    ]
                )

            instructions.append(instruction)
            instructions.append(
                {
                    "debug": [
                        (
                            "vcompare",
                            self.scratch[f"acc_{2*i}"],
                            [(0, 16 * i + n, "val") for n in range(8)],
                        ),
                        (
                            "vcompare",
                            self.scratch[f"acc_{2*i+1}"],
                            [(0, 16 * i + 8 + n, "val") for n in range(8)],
                        ),
                        (
                            "vcompare",
                            self.scratch[f"index_{2*i}"],
                            [(0, 16 * i + n, "idx") for n in range(8)],
                        ),
                        (
                            "vcompare",
                            self.scratch[f"index_{2*i+1}"],
                            [(0, 16 * i + 8 + n, "idx") for n in range(8)],
                        ),
                        (
                            "vcompare",
                            self.scratch[f"node_{2*i}"],
                            [(0, 16 * i + n, "node_val") for n in range(8)],
                        ),
                        (
                            "vcompare",
                            self.scratch[f"node_{2*i+1}"],
                            [(0, 16 * i + 8 + n, "node_val") for n in range(8)],
                        ),
                    ]
                }
            )

        op1r = self.alloc_scratch("op1r", 8)
        op3r = self.alloc_scratch("op3r", 8)
        VECTORIZED_HASH_STAGES = [
            ("+", hash_stage0_v, "+", "<<", twelve_v),
            ("^", hash_stage1_v, "^", ">>", nineteen_v),
            ("+", hash_stage2_v, "+", "<<", five_v),
            ("+", hash_stage3_v, "^", "<<", nine_v),
            ("+", hash_stage4_v, "+", "<<", three_v),
            ("^", hash_stage5_v, "^", ">>", sixteen_v),
        ]
        for r in range(rounds):
            # For each section...
            for i in range(16):
                # Calculate vectorized XOR.
                instructions.append(
                    {
                        "valu": [
                            (
                                "^",
                                self.scratch[f"acc_{i}"],
                                self.scratch[f"acc_{i}"],
                                self.scratch[f"node_{i}"],
                            ),
                        ]
                    }
                )

                # Calculate the hash using SIMD.
                for hi, (op1, vec1, op2, op3, vec3) in enumerate(VECTORIZED_HASH_STAGES):
                    instructions.append(
                        {
                            "valu": [
                                (
                                    op1,
                                    op1r,
                                    self.scratch[f"acc_{i}"],
                                    vec1,
                                ),
                                (
                                    op3,
                                    op3r,
                                    self.scratch[f"acc_{i}"],
                                    vec3,
                                ),
                            ]
                        }
                    )
                    instructions.append(
                        {
                            "valu": [
                                (
                                    op2,
                                    self.scratch[f"acc_{i}"],
                                    op1r,
                                    op3r,
                                ),
                            ]
                        }
                    )
                    instructions.append(
                        {
                            "debug": [
                                (
                                    "vcompare",
                                    self.scratch[f"acc_{i}"],
                                    [(0, 8 * i + n, "hash_stage", hi) for n in range(8)],
                                ),
                            ]
                        }
                    )

                # TODO: vswitch on the hashed value parity.


                # TODO: Load the next node values.
                pass

            # TODO: Do more than one round.
            break

        # TODO: Store the final answers back into memory.

        self.instrs.extend(instructions)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


class KernelBuilderOriginal:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(
                ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
            )

        return slots

    def build_kernel_original(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                )
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                )
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(
                    ("debug", ("compare", tmp_node_val, (round, i, "node_val")))
                )
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                )
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on pause {i+1}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)

    def test_kernel_prints(self):
        do_kernel_test(10, 16, 256, trace=True, prints=True)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
