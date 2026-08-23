#!/usr/bin/env node
/* Run the MythicGPT kernel screens and report paired numerical differences. */

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
    const argmax = values => values.reduce(
        (best, value, index) => Number(value) > Number(values[best]) ? index : best,
        0
    );
    return {
        same_length: true,
        exact,
        max_abs: maxAbs,
        mean_abs: sumAbs / baseline.length,
        total_variation: sumAbs / 2,
        same_argmax: argmax(baseline) === argmax(candidate)
    };
};

const main = async () => {
    const projectPath = process.argv[2];
    if (!projectPath) {
        throw new Error('Usage: run_mythic_kernel_vm.cjs <mythic-kernel-suite.sb3>');
    }

    const vm = new VirtualMachine();
    vm.setTurboMode(true);
    await vm.loadProject(fs.readFileSync(projectPath));
    vm.start();

    const completion = new Promise((resolve, reject) => {
        const timeout = setTimeout(
            () => reject(new Error('MythicGPT kernel suite timed out after 15 minutes')),
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

    const dense = {};
    for (const shape of ['128x128', '128x384', '384x128']) {
        const prefix = `linear ${shape}`;
        const baseline = get(`${prefix} current_append output`);
        dense[shape] = Object.fromEntries([
            'current_replace', 'mythic_four', 'mythic_four_unrolled'
        ].map(variant => [
            variant,
            compare(baseline, get(`${prefix} ${variant} output`))
        ]));
    }

    const quantBaseline = get('quant float current output');
    const quantizedCompute = Object.fromEntries([
        'quant int current', 'quant int mythic four',
        'quant int mythic four unrolled'
    ].map(variant => [
        variant.replaceAll(' ', '_'),
        compare(quantBaseline, get(`${variant} output`))
    ]));

    const softmax = {};
    for (const size of [32, 128, 192]) {
        const baseline = get(`softmax ${size} exact output`);
        softmax[size] = {
            stored_exact: compare(baseline, get(`softmax ${size} stored_exact output`)),
            mythic_lookup: compare(baseline, get(`softmax ${size} mythic_lookup output`))
        };
    }

    const expectedIntegers = Array.from(
        {length: 49152}, (_, index) => ((index * 29) % 255) - 127
    );
    const decode = {
        mythic_int8: compare(expectedIntegers, get('decode mythic int8 output')),
        cattorch_f16_length: get('decode cattorch f16 output').length
    };

    process.stdout.write(`${JSON.stringify({
        project: path.resolve(projectPath),
        turbo_mode: vm.runtime.turboMode,
        wall_seconds: wallSeconds,
        results: results.value.map(String),
        comparisons: {dense, quantized_compute: quantizedCompute, softmax, decode}
    }, null, 2)}\n`);
    vm.quit();
};

main().catch(error => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
