# Scratch interface

[Documentation home](index.md)

Scratch projects use an exported sprite through its blocks, lists, variables,
and broadcasts. This page lists every one of them, the names that
[projects](#projects) and [benchmark projects](#benchmark-projects) add, and
every value `cattorch status` can take. Together they're a versioned
contract:

- In an exported sprite, a name is public when it starts with `cattorch `
  (with the space), or is one of a model's `input` and `output` lists
  (`input`, `input_1`, ..., `output`, `output_1`, ...). Every other name in a
  sprite starts with `_` and is internal: weights, working lists, helper blocks. Internal names can
  change in any release, so don't use them in your scripts.
- Within interface version 1, which covers cattorch 1.x, no public name or
  status value is renamed or removed, and its meaning doesn't change. New
  names may be added.
- Each sprite has a `cattorch interface version` variable holding the version
  it implements, `1`. A sprite without it is from before cattorch 1.0. The
  version changes only when a name or status value is renamed or removed, or
  changes meaning, which no 1.x release does; adding names doesn't change it.

In the tables, a name in capitals stands for part of a name the export
chooses: ENTRY for an entry point, INPUT and OUTPUT for its lists, STATE for
a piece of state, NAME for a sprite in a project, CASE and VARIANT for a
benchmark case and its variant.

## How every block behaves

- Blocks take no arguments and run without screen refresh. A block reads
  its input lists and writes its output lists in the sprite.
- Call a sprite's blocks from one script at a time; see
  [overlapping calls](#overlapping-calls).
- Every block sets `cattorch status` when it ends: `ok`, or why it failed,
  from the [status table](#status-values). It then does a
  `broadcast and wait` of its completion broadcast, `BLOCK complete` (for
  `cattorch forward`, `cattorch forward complete`), whatever happened:
  success, invalid input, or another call running during it. Only a block
  that is stopped, by the stop sign or a `stop` block, sends none and leaves
  `cattorch status` as it was.
- So a script that reacts to `when I receive [cattorch forward complete]`
  never waits forever, and checks `cattorch status` before reading the
  outputs. A broadcast doesn't say which call sent it: a script that needs
  the result of its own call runs the block itself, which returns when the
  block ends, and reads `cattorch status` right after.
- A block that fails writes none of its output lists and changes no state:
  the outputs keep the values of the last call that succeeded. A block never
  changes its input lists. (A block taken over while it copies its results is
  the exception; see [overlapping calls](#overlapping-calls).)
- A block checks its input lists before it runs: the number of items, and
  each item. A program checks only the number of items of its
  [state lists](#program-sprites). A list of floating-point values must hold numbers (`Infinity`
  and `-Infinity` count; text, an empty item and `NaN` don't); a list of
  integers must hold whole numbers from -2^53 to 2^53, the ones a Scratch
  number holds exactly (`Infinity` doesn't count). Token IDs must be from 0 to the
  vocabulary size minus 1, and an integer input the model uses as an index
  must hold indices valid for every operation that uses it, as in PyTorch:
  from 0 to the size minus 1, or, where PyTorch counts a negative index from
  the end, as in `x[ids]`, from minus the size; see
  [index inputs](supported-models.md#index-inputs). An index the model
  computes from its inputs, such as `x[ids + 1]`, is checked as the block
  runs, and one out of range fails the block with `invalid item: LIST`
  naming the input it comes from. A bool list takes any values.
- A block decodes the stored weights itself when they aren't decoded yet,
  which is `cattorch init` running inside it: it sends
  `cattorch init complete` too. A program's entry point decodes only its own
  weights, which isn't an init.
- `cattorch prepare for save` clears the decoded weights and internal working
  lists, so a project saved after running stays small, and sets
  `cattorch initialized` to `0`; the next block decodes the weights again.
  Public lists keep their values, except that a generation sprite also
  resets, as `cattorch reset` does: its cache can't be kept.

## Status values

`cattorch status` starts empty, before any block has ended, and each block
sets it to one of these when it ends. LIST is the public list at fault.

| Status | Meaning |
|---|---|
| `ok` | The block succeeded. |
| `wrong number of items: LIST` | A fixed-size list doesn't hold its number of items, a variable-length list doesn't hold a whole number of rows, a [long list](#long-lists)'s shards don't each hold their part, or `cattorch decode` didn't get exactly one token. |
| `too few items: LIST` | A variable-length list holds fewer rows than its minimum, or `cattorch prefill` got no tokens. |
| `too many items: LIST` | A variable-length list holds more rows than its maximum, or `cattorch prefill` got more tokens than `cattorch max context`. |
| `invalid item: LIST` | An item isn't a number, or isn't a whole number in range, such as a token ID outside the vocabulary. |
| `state full: LIST` | An entry point's call would append past a state list's capacity. |
| `context full` | `cattorch decode` found `cattorch cache length` at `cattorch max context`. |
| `another call ran during this one` | Another of the sprite's blocks started while this one ran; see [overlapping calls](#overlapping-calls). |
| `an earlier call was stopped or is still running; reset first` | A block that changes the cache or states was stopped or taken over part way, or is still running. |

## Overlapping calls

Call a sprite's blocks from one script at a time, and let each block end
before another script starts one of the same sprite's blocks.

The blocks run without screen refresh, but scratch-vm doesn't let such a
block run forever: once it has run for about 500 ms, scratch-vm can pause it
at a loop repeat or a My Block call, redraw the screen, and run other scripts
before it continues. If one of those scripts starts another block of the
same sprite, the two calls overlap. They share the sprite's working lists,
its decoded weights and its states, so each can spoil the other's results.
The stop sign, or a `stop` block, can also stop a block part way.

The sprite detects most of this and reports it in `cattorch status`:

- **Taken over.** A block that another of the sprite's blocks started during
  writes none of its outputs, reports `another call ran during this one`,
  and still sends its completion broadcast. If it had itself started while
  an older block was paused, that older block fails the same way when it
  ends. A block taken over while it copies its results to its output lists
  leaves them mixed: a model, program or generation sprite then empties them
  (for a generator, `cattorch logits` and the top-k lists); a tokenizer
  leaves them, so its status is the only sign.
- **Weights.** A `cattorch init`, or a block decoding the weights itself,
  that another block starts during leaves the weights marked undecoded
  (`cattorch initialized` is `0`), and the next block decodes them again.
  `cattorch prepare for save` makes a block running during it fail.
- **Interrupted states.** A sprite with state (the cache of a generator, the
  states of a recurrent generator or a program) can be left with its states
  half written. Until it is cleared, some blocks are refused: they change
  nothing and report `an earlier call was stopped or is still running;
  reset first`. They are also refused while a block that writes the states
  is running in another script. `cattorch init` doesn't clear the condition.

|   | Model, tokenizer | Program | Generation, recurrent generation |
|---|---|---|---|
| Left interrupted by | Nothing: no states. | `cattorch reset`, or an entry point that updates state, stopped or taken over while it writes the states. | `cattorch reset`, `cattorch prefill` or `cattorch decode` stopped or taken over at any point. |
| Refused | Nothing. | Every entry point, in a program with state. An entry point of a program without state is never refused. | `cattorch decode`. |
| A refusal while another block runs | | Leaves the running block alone: it carries on and sets its own status when it ends. | Makes the running block fail: it skips the rest of its work and reports `another call ran during this one`. |
| Cleared by | | `cattorch reset`, which is never refused. | `cattorch reset` or `cattorch prefill`, which are never refused. |

Two cases are not detected, and are why the rule above matters:

- If a block that was taken over resumes while the newer block is itself
  paused, and hasn't ended when the newer one ends, or is stopped after it
  resumed, the newer block can report `ok` with results computed from
  working lists the old one changed.
- A block taken over while it decodes the weights carries on decoding when
  it resumes. A block that runs meanwhile can report `ok` with weights the
  old block half overwrote. The next block after the old one ends decodes
  them again.

Also, while a completion broadcast runs its listeners, Scratch can resume
another script, so a listener, or the script that ran the block, can read
another call's status instead of its own.

To recover, run the block again from one script. If the status is `an
earlier call was stopped or is still running; reset first`, run
`cattorch reset` first (in a generator, `cattorch prefill` also works): a
program's states go back to their initial values, and a generator starts
over from an empty cache or its initial states. To start clean after any overlap, run
`cattorch init`, which decodes the weights again, and then `cattorch reset`
in a sprite with state.

## Model sprites

A sprite from [`transpile`](getting-started.md) with an example input:

| Name | Kind | Meaning |
|---|---|---|
| `cattorch forward` | block | Runs the model on the input lists and writes the output lists. |
| `cattorch init` | block | Decodes the stored weights. Optional: `cattorch forward` decodes them when they aren't. |
| `cattorch prepare for save` | block | Clears the decoded weights and working lists before saving the project. |
| `input`, `input_1`, ... | list | The model's inputs, flattened in PyTorch order. |
| `output`, `output_1`, ... | list | The model's outputs, flattened the same way. |
| `cattorch status` | variable | How the last block ended; see [status values](#status-values). |
| `cattorch initialized` | variable | `1` once the weights are decoded, `0` before and after `cattorch prepare for save`. |
| `cattorch init complete`, `cattorch forward complete`, `cattorch prepare for save complete` | broadcast | Sent when each block ends. |

`TranspileResult.inputs` and `outputs` record each list's tensor shape. A
fixed-size list holds exactly that many items; a
[variable-length](supported-models.md#variable-length-inputs) one holds any
whole number of rows within its bounds.

## Generation sprites

A sprite from a [`GenerationProgram`](generation.md#scratch-interface):

| Name | Kind | Meaning |
|---|---|---|
| `cattorch init` | block | Decodes the stored weights. Optional: the other blocks decode them when they aren't. |
| `cattorch reset` | block | Empties the cache, `cattorch logits` and the top-k lists. |
| `cattorch prefill` | block | Resets, reads the prompt in `cattorch token ids` into the cache, and writes the next token's logits. |
| `cattorch decode` | block | Adds the one token in `cattorch token ids` to the cache and writes the next token's logits. |
| `cattorch prepare for save` | block | Clears the decoded weights and working lists, and resets. |
| `cattorch token ids` | list | Zero-based token IDs to read: the prompt, or one token. |
| `cattorch logits` | list | The next token's logit for every token ID. |
| `cattorch top k values`, `cattorch top k ids` | list | With `top_k`, the largest logits, largest first, and their token IDs. |
| `cattorch cache length` | variable | Tokens in the cache; `0` after a reset. |
| `cattorch max context` | variable | The most tokens the cache holds. |
| `cattorch status` | variable | How the last block ended; see [status values](#status-values). |
| `cattorch initialized` | variable | `1` once the weights are decoded, `0` before and after `cattorch prepare for save`. |
| `cattorch init complete`, `cattorch reset complete`, `cattorch prefill complete`, `cattorch decode complete`, `cattorch prepare for save complete` | broadcast | Sent when each block ends. |

A sprite from a [`RecurrentGenerationProgram`](recurrent-generation.md#scratch-interface)
has the same names. It has no cache: `cattorch reset` restores the model's
initial state, `cattorch cache length` counts the tokens read since then, and
`cattorch max context` is `max_context`, or `Infinity` when it has no limit.

## Program sprites

A sprite from an [`ExportProgram`](programs-and-moe.md#named-entry-points-and-state)
or [`@entrypoint` markers](markers.md) has a block per entry point, named
after it:

| Name | Kind | Meaning |
|---|---|---|
| `cattorch ENTRY` | block | Calls the entry point: reads its input lists and the states it takes (with markers, the ones its method reads or writes), checks their sizes, and writes its output lists and the states it updates. Other states are left alone. |
| `cattorch init` | block | Decodes every entry point's weights. Optional. |
| `cattorch reset` | block | Restores every state to its initial value. |
| `cattorch prepare for save` | block | Clears the decoded weights and working lists before saving the project. |
| `cattorch ENTRY INPUT`, `cattorch ENTRY OUTPUT` | list | An entry point's inputs and outputs, named after its parameters and outputs. |
| `cattorch state STATE` | list | A piece of persistent state. It belongs to the sprite: read it, or save and restore values the sprite wrote. A block checks its length, not its items, so writing anything else gives undefined results. |
| `cattorch status` | variable | How the last block ended; see [status values](#status-values). |
| `cattorch initialized` | variable | `1` once every entry point's weights are decoded, by `cattorch init` or by calls; `0` before and after `cattorch prepare for save`. |
| `cattorch ENTRY complete`, `cattorch init complete`, `cattorch reset complete`, `cattorch prepare for save complete` | broadcast | Sent when each block ends. |

## Tokenizer sprites

A sprite from [`transpile_tokenizer`](tokenizers.md):

| Name | Kind | Meaning |
|---|---|---|
| `cattorch tokenize` | block | Reads `cattorch text` and writes its token IDs to `cattorch token ids`. |
| `cattorch detokenize` | block | Reads `cattorch token ids` and writes the text to `cattorch text`. |
| `cattorch text` | variable | The text to tokenize, or the detokenized text. Starts empty. |
| `cattorch token ids` | list | Zero-based token IDs. |
| `cattorch status` | variable | How the last block ended; see [status values](#status-values). |
| `cattorch tokenize complete`, `cattorch detokenize complete` | broadcast | Sent when each block ends. |

## Every sprite

| Name | Kind | Meaning |
|---|---|---|
| `cattorch interface version` | variable | The interface version the sprite implements, `1`. |

Model, generation and program sprites also describe how their weights are
stored, in read-only variables:

| Name | Kind | Values |
|---|---|---|
| `cattorch storage compression` | variable | `base92` (packed into text), `base92-huffman-int4` (4-bit weights packed with a Huffman code), or `none` (one list item per weight). |
| `cattorch storage precision` | variable | `float32`, `float16`, `int8`, `int6`, or `int4`. |
| `cattorch storage scale precision` | variable | `float32` or `float16`: the precision of integer weights' scales. |
| `cattorch quantization method` | variable | `none`, `symmetric`, or `gptq`. |
| `cattorch quantization bits` | variable | `0` without quantization, or `4`, `6`, or `8`. |

## Long lists

A public list over 200,000 items is split into lists of at most 200,000:
`LIST`, `LIST shard 2`, `LIST shard 3`, and so on, which are public too. Two
variables describe it: `cattorch BASE shard count` and
`cattorch BASE logical length`, where BASE is the list's name without the
`cattorch ` it may start with. So `output` has `cattorch output shard count`,
and `cattorch logits` has `cattorch logits shard count`. See
[list sharding](storage.md#list-sharding).

The items run in order through the shards: every shard but the last one
holding items holds exactly 200,000, so a fixed-size list of 200,010 items
is 200,000 in `LIST` and 10 in `LIST shard 2`. A block checks each shard of
an input or state list it reads, and reports one laid out otherwise as
`wrong number of items: LIST`, and an invalid item in any shard as
`invalid item: LIST`, LIST being the first list's name.

Token IDs and indices are zero-based, as in PyTorch, though Scratch list
positions start at 1.

## Projects

A project from [`build_project`](projects-and-cli.md#bridges) puts each
model's and tokenizer's operations behind a bridge: a message that copies the
project's global lists into the sprite, runs the block, sets `NAME status`,
and, when it is `ok`, copies the outputs back. The global names start with
the sprite's name, NAME, then the sprite's own name for the list or variable
without the `cattorch ` it may start with. Each bridge is a block in the
sprite, `cattorch bridge MESSAGE`, which ends with a `broadcast and wait` of
`MESSAGE complete`.

| Name | Kind | Meaning |
|---|---|---|
| `NAME run` | broadcast | A model: runs `cattorch forward`. |
| `NAME predict` | broadcast | A classifier (with `labels`): writes the label of the largest output to `NAME prediction`. |
| `NAME ENTRY` | broadcast | A program: calls `cattorch ENTRY`. |
| `NAME prefill`, `NAME decode` | broadcast | A text generator: runs the block, then samples the next token. |
| `NAME tokenize`, `NAME detokenize` | broadcast | A tokenizer: runs the block. |
| `NAME run complete`, `NAME predict complete`, `NAME ENTRY complete`, `NAME prefill complete`, `NAME decode complete`, `NAME tokenize complete`, `NAME detokenize complete` | broadcast | Sent when each bridge ends. |
| `cattorch bridge NAME run`, `cattorch bridge NAME predict`, `cattorch bridge NAME ENTRY`, `cattorch bridge NAME prefill`, `cattorch bridge NAME decode`, `cattorch bridge NAME tokenize`, `cattorch bridge NAME detokenize` | block | The bridges, in the model's or tokenizer's sprite. |
| `NAME input`, `NAME output`, ... | list | A model's input and output lists. A list split into [shards](#long-lists) is bridged shard by shard, to `NAME output`, `NAME output shard 2`, and so on. |
| `NAME ENTRY INPUT`, `NAME ENTRY OUTPUT` | list | A program's entry point lists. |
| `NAME labels` | list | A classifier's labels, one per output. |
| `NAME token ids` | list | A text generator's prompt or one token, and a tokenizer's token IDs. |
| `NAME status` | variable | The sprite's `cattorch status` after its last bridge. |
| `NAME prediction` | variable | A classifier's predicted label. |
| `NAME next token` | variable | The token a text generator sampled, zero-based. |
| `NAME temperature` | variable | The sampling temperature; 0 picks the likeliest token. |
| `NAME text` | variable | A tokenizer's text. |

The project's Demo sprite has only internal names, such as the
`_cattorch demo prompt tokens` block.

## Benchmark projects

A project from a [benchmark builder](verification-and-benchmarking.md#scratch-benchmark-projects)
records how it ran and its timings on the stage, where
[`analyze_benchmark`](api-reference.md#analyze_benchmark) reads them. These
stage names are public:

| Name | Kind | Meaning |
|---|---|---|
| `cattorch exact timings`, `cattorch fast timings` | list | `build_paired_benchmark`: the seconds each timed call of a variant took. |
| `cattorch benchmark results` | list | The other builders: `CASE VARIANT: SECONDS` per case and variant, the time its timed calls took together (`CASE VARIANT init: SECONDS` and `CASE VARIANT forward: SECONDS` in a storage benchmark). |
| `cattorch storage sizes` | list | `build_storage_benchmark_suite`: `CASE VARIANT: BYTES bytes`, the `.sprite3` size of each case and storage mode. |
| `cattorch benchmark repeats`, `cattorch benchmark warmups` | variable | The timed and untimed calls per variant. |
| `cattorch benchmark comparison` | variable | The variant compared with the baseline, such as `fast` or `cached`; not in a storage benchmark. |

Everything else in a benchmark project is internal: the stage's scripts and
messages, and the sprites, one per case and variant (`CASE VARIANT`, or
`cattorch exact` and `cattorch fast` in a paired benchmark), whose own
interface is that of their kind of sprite.

Related: [getting started](getting-started.md), [glossary](glossary.md).
