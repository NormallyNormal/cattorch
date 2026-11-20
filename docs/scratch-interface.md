# Scratch interface

[Documentation home](index.md)

Scratch projects use an exported sprite through its broadcasts, global lists
and variables, and a few variables of its own. This page lists every one of
them, the names that [projects](#projects) and
[benchmark projects](#benchmark-projects) add, and every value `NAME status`
can take. Together they're a versioned contract.

## Using a sprite without opening it

You never need to select or open an exported sprite to use it. A large
model sprite makes the Scratch editor slow while it's selected, since the
editor shows its blocks, so leave it alone and work from another sprite or
the stage:

1. Put the sprite in your project. Scratch selects a sprite you upload
   (Sprite menu, **Upload Sprite**), so the editor shows all its blocks
   once, which can make it freeze or crash for a large model. For a large
   sprite, add it from Python with
   [`add_to_project`](projects-and-cli.md#add-sprites-to-your-own-project)
   or from the terminal with `cattorch add`, which write your `.sb3` with
   the sprite after your own sprites: the project opens with your first
   sprite selected, and the model's never is. A small sprite you can
   upload, then select another sprite, or the stage, right away.
2. In your own scripts, fill the sprite's input lists, then
   `broadcast [NAME forward] and wait`.
3. Read `NAME status`. When it's `ok`, the outputs are in the output lists;
   otherwise it says why the call failed.

NAME is the export's name: the `name` you passed to `transpile` (or
`transpile_tokenizer`, or a tokenizer's `save`), by default the output file
name without `.sprite3`. It's also the sprite's name in Scratch.

`broadcast and wait`, then reading `NAME status`, is the way to call a
sprite: it returns when the call ends, and the status is the call's own
(see [overlapping calls](#overlapping-calls) for the exceptions). Each call
also sends a completion broadcast, `NAME forward complete`, for scripts that
react to events instead of waiting.

The rule for names:

- **Global:** what a project reads or writes to use the sprite: its input,
  output and state lists, and the variables that change as it runs
  (`NAME status`, a generator's `NAME cache length`, a tokenizer's
  `NAME text`). Each is named NAME, a space, and what it is. The sprite
  uses them without defining them: when you upload it, Scratch creates each
  one on the stage, or uses the global of that name your project already
  has. Scratch creates a variable holding 0 and an empty list, so
  `NAME status` reads `0` until a call ends, and a program's state lists
  are empty until its first call or `NAME init` fills them.
  `add_to_project` declares them the same way, except that a new
  `NAME status` and `NAME text` start empty.
- **The sprite's own:** what is fixed when it's exported (its interface
  version, how its weights are stored, a generator's
  `cattorch max context`, [shard counts](#long-lists)) and
  `cattorch initialized`. Other sprites read them with the
  `([cattorch interface version] of [NAME])` sensing block.
- **Internal:** everything else in the sprite starts with `_`: weights,
  working lists, its blocks. Internal names can change in any release, so
  don't use them.
- **Messages:** every call is a broadcast, `NAME OP`, and its completion
  broadcast `NAME OP complete`. Every model, generation and program
  sprite also answers `cattorch prepare for save`, so that one
  `broadcast [cattorch prepare for save] and wait` prepares a whole project
  for saving.

Two sprites of one NAME share their globals and both answer the same
broadcasts, so a call runs in both and they overwrite each other's outputs.
That includes two uploads of the same file: Scratch renames the second
sprite (`NAME2`), but its globals and messages keep NAME. To use two copies
of one model, export them with different `name`s. NAMEs whose broadcasts
Scratch can't tell apart (it matches them ignoring case) are refused at
export, as are the names Scratch keeps for its menus (`_mouse_`,
`_stage_`, `_edge_`, `_myself_`, `_random_`) and a NAME whose broadcast
would be `cattorch prepare for save`.

The interface is versioned:

- Within interface version 1, which covers cattorch 1.x, no public name or
  status value is renamed or removed, and its meaning doesn't change. New
  names may be added.
- Each sprite has a `cattorch interface version` variable holding the
  version it implements, `1`. A sprite without it is from before cattorch
  1.0. The version changes only when a name or status value is renamed or
  removed, or changes meaning, which no 1.x release does; adding names
  doesn't change it.

In the tables, a name in capitals stands for part of a name the export
chooses: NAME for the export, ENTRY for an entry point, INPUT and OUTPUT for
its lists, STATE for a piece of state, CASE and VARIANT for a benchmark case
and its variant. The kinds are a *broadcast*, a global *list* or
*variable*, and a *sprite variable*, the sprite's own.

## How every call behaves

- A call is a broadcast that takes no arguments; the sprite runs it
  without screen refresh. It reads its input lists and writes its output
  lists.
- Call a sprite from one script at a time; see
  [overlapping calls](#overlapping-calls).
- When a call starts, `NAME status` becomes `not finished`. When it ends,
  `NAME status` says how: `ok`, or why it failed, from the
  [status table](#status-values). Then the call does a `broadcast and wait`
  of its completion broadcast, `NAME OP complete` (for `NAME forward`,
  `NAME forward complete`), whatever happened: success, invalid input, or
  another call running during it. A call that is stopped, by the stop sign
  or a `stop` block, or [restarted](#overlapping-calls), sends none and
  leaves `NAME status` at `not finished`.
- So a script that reacts to `when I receive [NAME forward complete]` never
  waits forever, and checks `NAME status` before reading the outputs. A
  broadcast doesn't say which call sent it: a script that needs the result
  of its own call does `broadcast [NAME forward] and wait`, which returns
  when the call ends, and reads `NAME status` right after.
- A call that fails writes none of its output lists and changes no state:
  the outputs keep the values of the last call that succeeded. A call never
  changes its input lists. (A call taken over, stopped or restarted while
  it copies its results is the exception; see
  [overlapping calls](#overlapping-calls).)
- A call checks its input lists before it runs: the number of items, and
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
  computes from its inputs, such as `x[ids + 1]`, is checked as the call
  runs, and one out of range fails the call with `invalid item: LIST`
  naming the input it comes from. A bool list must hold `true` and `false`
  as boolean blocks such as `<(a) > (b)>` give them, or the numbers `1` and
  `0` (also as text, as typed into a list monitor). The text `true` or
  `false` is refused: arithmetic reads it as 0, but it isn't 0, so operations
  would disagree on what it means; type `1` or `0` instead.
- A call decodes the stored weights itself when they aren't decoded yet,
  as `NAME init` does, and sends only its own completion broadcast. A
  program's entry point decodes only its own weights.
- `NAME prepare for save` (or the shared `cattorch prepare for save`)
  clears the decoded weights and every internal working list, so a project
  saved after running holds no more internal data than the exported
  sprite, and sets `cattorch initialized` to `0`; the next call decodes the
  weights again. Global lists keep their values, except that a generation
  sprite also resets, as `NAME reset` does: its cache can't be kept.

## Status values

`NAME status` is `not finished` while a call runs, and each call sets it to
one of the others when it ends. LIST is the global list at fault, such as
`points input`.

| Status | Meaning |
|---|---|
| `ok` | The call succeeded. |
| `not finished` | A call has started and hasn't ended: it is still running, or it was stopped, or a second broadcast of its message restarted it; see [overlapping calls](#overlapping-calls). |
| `wrong number of items: LIST` | A fixed-size list doesn't hold its number of items, a variable-length list doesn't hold a whole number of rows, a [long list](#long-lists)'s shards don't each hold their part, or `NAME decode` didn't get exactly one token. |
| `too few items: LIST` | A variable-length list holds fewer rows than its minimum, or `NAME prefill` got no tokens. |
| `too many items: LIST` | A variable-length list holds more rows than its maximum, or `NAME prefill` got more tokens than `cattorch max context`. |
| `invalid item: LIST` | An item isn't a number, or isn't a whole number in range, such as a token ID outside the vocabulary. |
| `state full: LIST` | An entry point's call would append past a state list's capacity. |
| `context full` | `NAME decode` found `NAME cache length` at `cattorch max context`. |
| `another call ran during this one` | Another of the sprite's calls started while this one ran; see [overlapping calls](#overlapping-calls). |
| `an earlier call was stopped or is still running; reset first` | A call that changes the cache or states was stopped or taken over part way, or is still running. |

## Overlapping calls

Call a sprite from one script at a time, and let each call end before
another script starts one of the same sprite's calls.

The calls run without screen refresh, but scratch-vm doesn't let one run
forever: once it has run for about 500 ms, scratch-vm can pause it at a
loop repeat or a My Block call, redraw the screen, and run other scripts
before it continues. If one of those scripts starts another call of the
same sprite, the two calls overlap. They share the sprite's working lists,
its decoded weights and its states, so each can spoil the other's results.
The stop sign, or a `stop` block, can also stop a call part way.

The sprite detects most of this and reports it in `NAME status`:

- **Restarted.** A broadcast of a message whose script is running, such
  as a second `NAME forward` while one runs, makes Scratch stop that
  script and start it again: the first call ends where it was, as if
  stopped, and sends no completion broadcast. The script that broadcast
  it gets back from its `broadcast and wait` at once, while the second
  call runs, and reads `not finished`, never an earlier call's status.
  (If the second call ends first, it reads that call's status, as any
  script can read another call's.) The second call reports on its own.
- **Taken over.** A call that another of the sprite's calls started
  during (one of another message, such as `NAME init` during
  `NAME forward`) writes none of its outputs, reports
  `another call ran during this one`, and still sends its completion
  broadcast. If it had itself started while an older call was paused, that
  older call fails the same way when it ends. A call taken over while it
  copies its results to its output lists leaves them mixed: a model,
  program or generation sprite then empties them (for a generator,
  `NAME logits` and the top-k lists); a tokenizer leaves them, so its
  status is the only sign. A call stopped or restarted while it copies
  them leaves them mixed too.
- **Weights.** A `NAME init`, or a call decoding the weights itself, that
  another call starts during leaves the weights marked undecoded
  (`cattorch initialized` is `0`), and the next call decodes them again.
  `NAME prepare for save` makes a call running during it fail.
- **Interrupted states.** A sprite with state (the cache of a generator, the
  states of a recurrent generator or a program) can be left with its states
  half written. Until it is cleared, some calls are refused: they change
  nothing and report `an earlier call was stopped or is still running;
  reset first`. They are also refused while a call that writes the states
  is running in another script. `NAME init` doesn't clear the condition.

|   | Model, tokenizer | Program | Generation, recurrent generation |
|---|---|---|---|
| Left interrupted by | Nothing: no states. | `NAME reset`, or an entry point that updates state, stopped, restarted or taken over while it writes the states. | `NAME reset`, `NAME prefill` or `NAME decode` stopped, restarted or taken over at any point. |
| Refused | Nothing. | Every entry point, in a program with state. An entry point of a program without state is never refused. | `NAME decode`. |
| A refusal while another call runs | | Leaves the running call alone: it carries on and sets its own status when it ends. | Makes the running call fail: it skips the rest of its work and reports `another call ran during this one`. |
| Cleared by | | `NAME reset`, which is never refused. | `NAME reset` or `NAME prefill`, which are never refused. |

Two cases are not detected, and are why the rule above matters:

- If a call that was taken over resumes while the newer call is itself
  paused, and hasn't ended when the newer one ends, or is stopped after it
  resumed, the newer call can report `ok` with results computed from
  working lists the old one changed.
- A call taken over while it decodes the weights carries on decoding when
  it resumes. A call that runs meanwhile can report `ok` with weights the
  old call half overwrote. The next call after the old one ends decodes
  them again.

Also, while a completion broadcast runs its listeners, Scratch can resume
another script, so a listener, or the script that made the call, can read
another call's status instead of its own.

To recover, call again from one script. If the status is `an
earlier call was stopped or is still running; reset first`, broadcast
`NAME reset` first (in a generator, `NAME prefill` also works): a
program's states go back to their initial values, and a generator starts
over from an empty cache or its initial states. To start clean after any
overlap, broadcast `NAME init`, which decodes the weights again, and then
`NAME reset` in a sprite with state.

## Model sprites

A sprite from [`transpile`](getting-started.md) with an example input:

| Name | Kind | Meaning |
|---|---|---|
| `NAME forward` | broadcast | Runs the model on the input lists and writes the output lists. |
| `NAME init` | broadcast | Decodes the stored weights. Optional: `NAME forward` decodes them when they aren't. |
| `NAME prepare for save` | broadcast | Clears the decoded weights and working lists before saving the project. |
| `NAME input`, `NAME input_1`, ... | list | The model's inputs, flattened in PyTorch order. |
| `NAME output`, `NAME output_1`, ... | list | The model's outputs, flattened the same way. |
| `NAME status` | variable | How the last call ended; see [status values](#status-values). |
| `cattorch initialized` | sprite variable | `1` once the weights are decoded, `0` before and after prepare for save. |
| `NAME forward complete`, `NAME init complete`, `NAME prepare for save complete` | broadcast | Sent when each call ends. |

`TranspileResult.inputs` and `outputs` name each list and record its tensor
shape, and `TranspileResult.broadcasts` the calls. A fixed-size list holds
exactly that many items; a
[variable-length](supported-models.md#variable-length-inputs) one holds any
whole number of rows within its bounds.

## Generation sprites

A sprite from a [`GenerationProgram`](generation.md#scratch-interface):

| Name | Kind | Meaning |
|---|---|---|
| `NAME init` | broadcast | Decodes the stored weights. Optional: the other calls decode them when they aren't. |
| `NAME reset` | broadcast | Empties the cache, `NAME logits` and the top-k lists. |
| `NAME prefill` | broadcast | Resets, reads the prompt in `NAME token ids` into the cache, and writes the next token's logits. |
| `NAME decode` | broadcast | Adds the one token in `NAME token ids` to the cache and writes the next token's logits. |
| `NAME prepare for save` | broadcast | Clears the decoded weights and working lists, and resets. |
| `NAME token ids` | list | Zero-based token IDs to read: the prompt, or one token. |
| `NAME logits` | list | The next token's logit for every token ID. |
| `NAME top k values`, `NAME top k ids` | list | With `top_k`, the largest logits, largest first, and their token IDs. |
| `NAME cache length` | variable | Tokens in the cache; `0` after a reset. |
| `NAME status` | variable | How the last call ended; see [status values](#status-values). |
| `cattorch max context` | sprite variable | The most tokens the cache holds. |
| `cattorch initialized` | sprite variable | `1` once the weights are decoded, `0` before and after prepare for save. |
| `NAME init complete`, `NAME reset complete`, `NAME prefill complete`, `NAME decode complete`, `NAME prepare for save complete` | broadcast | Sent when each call ends. |

A sprite from a [`RecurrentGenerationProgram`](recurrent-generation.md#scratch-interface)
has the same names. It has no cache: `NAME reset` restores the model's
initial state, `NAME cache length` counts the tokens read since then, and
`cattorch max context` is `max_context`, or `Infinity` when it has no limit.

## Program sprites

A sprite from an [`ExportProgram`](programs-and-moe.md#named-entry-points-and-state)
or [`@entrypoint` markers](markers.md) has a broadcast per entry point,
named after it. Scratch matches broadcast names ignoring case, and list and
variable names exactly, so no two entry points' broadcasts are the same
message ignoring case; see the
[rules](programs-and-moe.md#named-entry-points-and-state).

| Name | Kind | Meaning |
|---|---|---|
| `NAME ENTRY` | broadcast | Calls the entry point: reads its input lists and the states it takes (with markers, the ones its method reads or writes), checks their sizes, and writes its output lists and the states it updates. Other states are left alone. |
| `NAME init` | broadcast | Decodes every entry point's weights, and fills state lists that are still empty. Optional. |
| `NAME reset` | broadcast | Restores every state to its initial value. |
| `NAME prepare for save` | broadcast | Clears the decoded weights and working lists before saving the project. |
| `NAME ENTRY INPUT`, `NAME ENTRY OUTPUT` | list | An entry point's inputs and outputs, named after its parameters and outputs. |
| `NAME state STATE` | list | A piece of persistent state. It belongs to the sprite: read it, or save and restore values the sprite wrote. A call checks its length, and its items only where the model uses the state as an index ([index inputs](supported-models.md#index-inputs)), so writing anything else gives undefined results. Empty when the sprite is uploaded, until the first call or `NAME init` fills it with its initial value; a project you save keeps it. |
| `NAME status` | variable | How the last call ended; see [status values](#status-values). |
| `cattorch initialized` | sprite variable | `1` once every entry point's weights are decoded, by `NAME init` or by calls; `0` before and after prepare for save. |
| `NAME ENTRY complete`, `NAME init complete`, `NAME reset complete`, `NAME prepare for save complete` | broadcast | Sent when each call ends. |

## Tokenizer sprites

A sprite from [`transpile_tokenizer`](tokenizers.md):

| Name | Kind | Meaning |
|---|---|---|
| `NAME tokenize` | broadcast | Reads `NAME text` and writes its token IDs to `NAME token ids`. |
| `NAME detokenize` | broadcast | Reads `NAME token ids` and writes the text to `NAME text`. |
| `NAME text` | variable | The text to tokenize, or the detokenized text. |
| `NAME token ids` | list | Zero-based token IDs. |
| `NAME status` | variable | How the last call ended; see [status values](#status-values). |
| `NAME tokenize complete`, `NAME detokenize complete` | broadcast | Sent when each call ends. |

A tokenizer doesn't answer `cattorch prepare for save`: it has nothing to
clear.

## Every sprite

| Name | Kind | Meaning |
|---|---|---|
| `cattorch interface version` | sprite variable | The interface version the sprite implements, `1`. |

Model, generation and program sprites also describe how their weights are
stored, in sprite variables:

| Name | Kind | Values |
|---|---|---|
| `cattorch storage compression` | sprite variable | `base92` (packed into text), `base92-huffman-int4` (4-bit weights packed with a Huffman code), or `none` (one list item per weight). |
| `cattorch storage precision` | sprite variable | `float32`, `float16`, `int8`, `int6`, or `int4`. |
| `cattorch storage scale precision` | sprite variable | `float32` or `float16`: the precision of integer weights' scales. |
| `cattorch quantization method` | sprite variable | `none`, `symmetric`, or `gptq`. |
| `cattorch quantization bits` | sprite variable | `0` without quantization, or `4`, `6`, or `8`. |

Every model, generation and program sprite answers one more message:

| Name | Kind | Meaning |
|---|---|---|
| `cattorch prepare for save` | broadcast | Runs the sprite's `NAME prepare for save`, which sends `NAME prepare for save complete`. It has no completion broadcast of its own: `broadcast and wait` returns once every sprite is done. |

## Long lists

A global list over 200,000 items is split into lists of at most 200,000:
`LIST`, `LIST shard 2`, `LIST shard 3`, and so on, which are global too,
such as `NAME output shard 2`. Two sprite variables describe it:
`cattorch BASE shard count` and `cattorch BASE logical length`, where BASE
is the list's name without NAME: `NAME output` has
`cattorch output shard count`, and `NAME logits` has
`cattorch logits shard count`. See [list sharding](storage.md#list-sharding).

The items run in order through the shards: every shard but the last one
holding items holds exactly 200,000, so a fixed-size list of 200,010 items
is 200,000 in `LIST` and 10 in `LIST shard 2`. A call checks each shard of
an input or state list it reads, and reports one laid out otherwise as
`wrong number of items: LIST`, and an invalid item in any shard as
`invalid item: LIST`, LIST being the first list's name.

Token IDs and indices are zero-based, as in PyTorch, though Scratch list
positions start at 1.

## Projects

A project from [`build_project`](projects-and-cli.md#projects) holds the
exported sprites as they are, named NAME, with their globals declared on
the stage (the input lists start from `example_inputs`, or zeros, and
`NAME status` and a tokenizer's `NAME text` start empty), and a small
Demo sprite, selected when the project opens. The Demo runs the project
(the green flag, the space key, the text generator's questions) and holds
the project's own messages:

| Name | Kind | Meaning |
|---|---|---|
| `NAME predict` | broadcast | A classifier (with `labels`): writes the label of the largest output to `NAME prediction`. |
| `NAME sample` | broadcast | A text generator: picks the next token from `NAME logits`, at `NAME temperature`, into `NAME next token`. Broadcast it after a `NAME prefill` or `NAME decode` that ended `ok`. |
| `NAME predict complete`, `NAME sample complete` | broadcast | Sent when each ends. |
| `NAME labels` | list | A classifier's labels, one per output. |
| `NAME prediction` | variable | A classifier's predicted label. |
| `NAME next token` | variable | The token a text generator sampled, zero-based. |
| `NAME temperature` | variable | The sampling temperature; 0 picks the likeliest token. |

The Demo sprite is named `Demo` (`Demo2` when an exported sprite is named
`Demo`) and has only internal names besides these, such as the
`_cattorch demo prompt tokens` block. `ProjectResult` lists the sprites,
every broadcast of the project, and its global lists and variables. No two
of a project's sprites, globals or broadcasts may be the same to Scratch:
`build_project` refuses a tokenizer named like its model, for one.

[`add_to_project`](projects-and-cli.md#add-sprites-to-your-own-project)
adds no names: it puts the exported sprites, as they are, after a project's
own sprites and declares their globals and messages on its stage, using the
ones the project already has. A project with no sprites also gets a blank
`Sprite1` (`Sprite2` when an exported sprite is named `Sprite1`), first, so
that it doesn't open on a model.

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

The project's sprites are exported sprites, one per case and variant, named
`CASE VARIANT` (`cattorch exact` and `cattorch fast` in a paired benchmark),
with the interface of their kind of sprite: the stage calls them with
their broadcasts, and their globals (`CASE VARIANT input`,
`CASE VARIANT output`, ...) are on the stage. Everything else in a
benchmark project is internal: the stage's scripts and its lists and
variables starting with `_`.

Related: [getting started](getting-started.md), [glossary](glossary.md).
