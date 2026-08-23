"""Build a focused append-versus-replace screen for Mythic's linear loop."""

from pathlib import Path

from build_mythic_kernel_suite import (
    _current_linear_program,
    _linear_names,
    _linear_values,
    _mythic_linear_program,
    _sprite_target,
)
from cattorch.benchmark import (
    SUITE_RESULTS,
    _StageBlocks,
    _add_broadcast_entry,
    _stage_target,
    _write_project,
)
from cattorch.transpiler import _add_warp_procedure, _merge_lists_by_name


OUTPUT = Path(__file__).parent / "artifacts" / "mythic_append_suite.sb3"


def main():
    programs = []
    cases = []
    for inner, outputs, iterations in (
        (128, 128, 80),
        (128, 384, 24),
        (384, 128, 24),
    ):
        input_name, weight_name = _linear_names(inner, outputs)
        input_values, weight_values = _linear_values(inner, outputs)
        stem = f"append_{inner}x{outputs}"

        for suffix, preallocated in (
            ("current_append", False),
            ("current_replace", True),
        ):
            procedure = f"{stem}_{suffix}"
            output_name = f"append {inner}x{outputs} {suffix} output"
            programs.append((procedure, _current_linear_program(
                procedure, inner, outputs,
                input_name=input_name,
                weight_name=weight_name,
                input_values=input_values,
                weight_values=weight_values,
                output_name=output_name,
                preallocated=preallocated,
            )))
            cases.append((procedure, iterations))

        unroll_options = (False, True) if inner == 384 else (False,)
        for unrolled in unroll_options:
            structure = "unrolled" if unrolled else "faithful"
            for destination, preallocated in (("append", False), ("replace", True)):
                suffix = f"mythic_{structure}_{destination}"
                procedure = f"{stem}_{suffix}"
                output_name = f"append {inner}x{outputs} {suffix} output"
                programs.append((procedure, _mythic_linear_program(
                    procedure, inner, outputs,
                    input_name=input_name,
                    weight_name=weight_name,
                    input_values=input_values,
                    weight_values=weight_values,
                    output_name=output_name,
                    unrolled=unrolled,
                    preallocated=preallocated,
                )))
                cases.append((procedure, iterations))

    sprite, sprite_md5ext, sprite_bytes = _sprite_target(
        "Mythic append versus replace screens"
    )
    for index, (procedure, program) in enumerate(programs):
        _add_warp_procedure(sprite, procedure, program, x=640, y=index * 100)
    _merge_lists_by_name(sprite, {
        entry[0]
        for entry in sprite["lists"].values()
        if entry[0].endswith((" input", " weight"))
    })

    broadcasts = {}
    for procedure, _ in programs:
        message = f"benchmark {procedure}"
        broadcasts[message] = f"benchmark_{procedure}"
        _add_broadcast_entry(sprite, message, broadcasts[message], procedure)

    result_id = "cattorch_mythic_append_results"
    builder = _StageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    specs = [("clear", SUITE_RESULTS)]
    for procedure, iterations in cases:
        message = f"benchmark {procedure}"
        specs.extend((
            ("broadcast", message),
            ("reset_timer",),
            ("repeat", iterations, (("broadcast", message),)),
            ("record_result", SUITE_RESULTS, f"{procedure} x{iterations}"),
        ))
    hat = builder._id()
    first, _ = builder._chain(tuple(specs), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked", "next": first, "parent": None,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
        "x": 0, "y": 0,
    }
    stage, stage_md5ext, stage_bytes = _stage_target(
        builder.blocks, broadcasts, {result_id: [SUITE_RESULTS, []]},
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 600, "height": 320, "x": 10, "y": 10, "visible": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = _write_project(
        OUTPUT, stage, [sprite], {sprite_md5ext: sprite_bytes},
        (stage_md5ext, stage_bytes), "Mythic append suite", monitors=[monitor],
    )
    print(f"Mythic append suite written: {path}")


if __name__ == "__main__":
    main()
