#!/usr/bin/env node
/* Time MythicGPT initialization and one token step in the official Scratch VM. */

'use strict';

const fs = require('fs');
const path = require('path');
const JSZip = require('jszip');
const VirtualMachine = require('@scratch/scratch-vm');
require('./node_modules/@scratch/scratch-vm/src/util/log').settings.minLevel = 6;

const waitForThreads = async (vm, timeoutMs) => {
    const start = Date.now();
    while (vm.runtime.threads.some(thread => thread.status !== 4)) {
        if (Date.now() - start > timeoutMs) {
            throw new Error('MythicGPT procedure timed out');
        }
        await new Promise(resolve => setTimeout(resolve, 5));
    }
};

const callBlock = procedure => ({
    opcode: 'procedures_call', next: null, parent: null,
    inputs: {}, fields: {}, shadow: false, topLevel: true, x: 0, y: 0,
    mutation: {
        tagName: 'mutation', children: [], proccode: procedure,
        argumentids: '[]', warp: 'true'
    }
});

const main = async () => {
    const projectPath = process.argv[2];
    if (!projectPath) {
        throw new Error('Usage: run_mythic_forward_vm.cjs <mythic-gpt.sb3>');
    }

    const zip = await JSZip.loadAsync(fs.readFileSync(projectPath));
    const project = JSON.parse(await zip.file('project.json').async('string'));
    const sprite = project.targets.find(target => target.name === 'AI');
    if (!sprite) throw new Error('Project contains no AI sprite');

    // Prevent UI hats from running while the model procedures are timed.
    for (const target of project.targets) {
        for (const [identifier, block] of Object.entries(target.blocks)) {
            if (block.opcode.startsWith('event_')) delete target.blocks[identifier];
        }
    }
    sprite.blocks.cattorch_bench_ld = callBlock('ld');
    sprite.blocks.cattorch_bench_fw = callBlock('fw');
    sprite.blocks.cattorch_bench_pick = callBlock('pick');
    zip.file('project.json', JSON.stringify(project));

    const vm = new VirtualMachine();
    vm.setTurboMode(true);
    await vm.loadProject(await zip.generateAsync({
        type: 'nodebuffer', compression: 'DEFLATE'
    }));
    vm.start();

    const ai = vm.runtime.targets.find(target => target.getName() === 'AI');
    const stage = vm.runtime.getTargetForStage();
    const run = async identifier => {
        const start = process.hrtime.bigint();
        vm.runtime._pushThread(identifier, ai, {stackClick: true});
        await waitForThreads(vm, 15 * 60 * 1000);
        return Number(process.hrtime.bigint() - start) / 1e9;
    };

    const initialization = await run('cattorch_bench_ld');
    for (const variable of Object.values(stage.variables)) {
        if (variable.name === 'k') variable.value = 3;
        if (variable.name === 't') variable.value = 1;
    }
    const hiddenForward = await run('cattorch_bench_fw');
    const outputHeadAndPick = await run('cattorch_bench_pick');

    const lists = Object.fromEntries(Object.values(stage.variables)
        .filter(variable => Array.isArray(variable.value))
        .map(variable => [variable.name, variable.value]));
    const report = {
        project: path.resolve(projectPath),
        turbo_mode: vm.runtime.turboMode,
        initialization_seconds: initialization,
        hidden_forward_seconds: hiddenForward,
        output_head_and_pick_seconds: outputHeadAndPick,
        generated_token_step_seconds: hiddenForward + outputHeadAndPick,
        decoded_lengths: Object.fromEntries([
            'E_0', 'E_1', 'E_2', 'P_0', 'qkv0_w_0', 'fdn0_w_0',
            'qkv4_w_0', 'fdn4_w_0', 'ln_f_w_0'
        ].map(name => [name, lists[name].length])),
        cache_lengths: Object.fromEntries(
            ['K0', 'V0', 'K1', 'V1', 'K2', 'V2',
                'K3', 'V3', 'K4', 'V4']
                .map(name => [name, lists[name].length])
        ),
        logits: lists.LG.length
    };
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    vm.quit();
};

main().catch(error => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
