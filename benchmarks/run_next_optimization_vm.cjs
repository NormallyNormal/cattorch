#!/usr/bin/env node
/* Run the next optimization suite and verify each benchmark-only candidate. */

'use strict';

const fs = require('fs');
const path = require('path');
const VirtualMachine = require('@scratch/scratch-vm');

const RESULTS_NAME = 'cattorch benchmark results';

const compare = (baseline, candidate) => {
    if (baseline.length !== candidate.length) {
        return {same_length: false, baseline_length: baseline.length,
            candidate_length: candidate.length};
    }
    let maxAbs = 0;
    let sumAbs = 0;
    let exact = true;
    for (let index = 0; index < baseline.length; index++) {
        const left = Number(baseline[index]);
        const right = Number(candidate[index]);
        const difference = Math.abs(left - right);
        maxAbs = Math.max(maxAbs, difference);
        sumAbs += difference;
        exact = exact && Object.is(left, right);
    }
    return {
        same_length: true,
        exact,
        max_abs: maxAbs,
        mean_abs: baseline.length ? sumAbs / baseline.length : 0
    };
};

const main = async () => {
    const projectPath = process.argv[2];
    if (!projectPath) {
        throw new Error('Usage: run_next_optimization_vm.cjs <suite.sb3>');
    }

    const vm = new VirtualMachine();
    vm.setTurboMode(true);
    await vm.loadProject(fs.readFileSync(projectPath));
    vm.start();
    const completion = new Promise((resolve, reject) => {
        const timeout = setTimeout(
            () => reject(new Error('Next optimization suite timed out after 15 minutes')),
            15 * 60 * 1000
        );
        vm.once('PROJECT_RUN_STOP', () => {
            clearTimeout(timeout);
            resolve();
        });
    });
    const wallStart = process.hrtime.bigint();
    vm.greenFlag();
    await completion;
    const wallSeconds = Number(process.hrtime.bigint() - wallStart) / 1e9;

    const stage = vm.runtime.getTargetForStage();
    const results = Object.values(stage.variables).find(
        variable => variable.name === RESULTS_NAME
    );
    const listMap = new Map(vm.runtime.targets.flatMap(target =>
        Object.values(target.variables)
            .filter(variable => Array.isArray(variable.value))
            .map(variable => [variable.name, variable.value])
    ));
    const get = name => {
        const values = listMap.get(name);
        if (!values) throw new Error(`Missing output list: ${name}`);
        return values;
    };
    const output = name => get(`${name} output`);

    const loopBaseline = output('loop_repeat_change');
    const loops = Object.fromEntries([
        'loop_for_each', 'loop_dynamic_repeat', 'loop_literal_repeat',
        'loop_unroll_1', 'loop_unroll_2', 'loop_unroll_4', 'loop_unroll_8'
    ].map(name => [name, compare(loopBaseline, output(name))]));

    const attention = {};
    for (const context of [1, 4, 16, 32, 64, 128]) {
        attention[context] = compare(
            output(`attention_c${context}_current`),
            output(`attention_c${context}_fused`)
        );
    }

    const comparisons = {
        loops,
        sharded_head: compare(
            output('head_shards_generic'), output('head_shards_direct')
        ),
        attention,
        interleaved_linear: compare(
            output('linear_grouped_current'), output('linear_grouped_interleaved')
        ),
        rms_linear: compare(
            output('rms_linear_current'), output('rms_linear_fused')
        ),
        swiglu: compare(output('swiglu_current'), output('swiglu_fused')),
        prefill_k_cache: compare(
            get('prefill_final_current K cache'),
            get('prefill_final_cache_only K cache')
        ),
        prefill_v_cache: compare(
            get('prefill_final_current V cache'),
            get('prefill_final_cache_only V cache')
        )
    };

    process.stdout.write(`${JSON.stringify({
        project: path.resolve(projectPath),
        turbo_mode: vm.runtime.turboMode,
        wall_seconds: wallSeconds,
        results: results.value.map(String),
        comparisons
    }, null, 2)}\n`);
    vm.quit();
};

main().catch(error => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
