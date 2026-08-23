#!/usr/bin/env node
/* Time one saved/decompressed Mochi forward_step in the official Scratch VM. */

'use strict';

const fs = require('fs');
const path = require('path');
const JSZip = require('jszip');
const VirtualMachine = require('@scratch/scratch-vm');
require('./node_modules/@scratch/scratch-vm/src/util/log').settings.minLevel = 6;

const main = async () => {
    const projectPath = process.argv[2];
    if (!projectPath) {
        throw new Error('Usage: run_mochi_forward_vm.cjs <mochi.sb3>');
    }

    const zip = await JSZip.loadAsync(fs.readFileSync(projectPath));
    const project = JSON.parse(await zip.file('project.json').async('string'));
    const sprite = project.targets.find(target => !target.isStage);
    if (!sprite) throw new Error('Project contains no sprite');

    const hats = Object.entries(sprite.blocks).filter(
        ([, block]) => block.opcode === 'event_whenflagclicked'
    );
    if (!hats.length) throw new Error('Project contains no green-flag script');
    for (const [identifier] of hats) delete sprite.blocks[identifier];

    const hatId = 'cattorch_mochi_benchmark_hat';
    const callId = 'cattorch_mochi_benchmark_call';
    sprite.blocks[hatId] = {
        opcode: 'event_whenflagclicked', next: callId, parent: null,
        inputs: {}, fields: {}, shadow: false, topLevel: true, x: 0, y: 0
    };
    sprite.blocks[callId] = {
        opcode: 'procedures_call', next: null, parent: hatId,
        inputs: {}, fields: {}, shadow: false, topLevel: false,
        mutation: {
            tagName: 'mutation', children: [], proccode: 'forward_step',
            argumentids: '[]', warp: 'true'
        }
    };

    zip.file('project.json', JSON.stringify(project));
    const benchmarkProject = await zip.generateAsync({
        type: 'nodebuffer', compression: 'DEFLATE'
    });
    const vm = new VirtualMachine();
    vm.setTurboMode(true);
    await vm.loadProject(benchmarkProject);
    vm.start();

    const completion = new Promise((resolve, reject) => {
        const timeout = setTimeout(
            () => reject(new Error('Mochi forward timed out after 15 minutes')),
            15 * 60 * 1000
        );
        vm.once('PROJECT_RUN_STOP', () => {
            clearTimeout(timeout);
            resolve();
        });
    });
    const start = process.hrtime.bigint();
    vm.greenFlag();
    await completion;
    const seconds = Number(process.hrtime.bigint() - start) / 1e9;

    const target = vm.runtime.targets.find(item => !item.isStage);
    const logits = Object.values(target.variables).find(
        variable => variable.name === 'logits'
    );
    let argmax = -1;
    let maximum = -Infinity;
    logits.value.forEach((raw, index) => {
        const value = Number(raw);
        if (value > maximum) {
            maximum = value;
            argmax = index;
        }
    });
    const report = JSON.stringify({
        project: path.resolve(projectPath),
        turbo_mode: vm.runtime.turboMode,
        forward_seconds: seconds,
        logits: logits.value.length,
        argmax,
        maximum
    }, null, 2);
    if (process.env.MOCHI_RESULT_PATH) {
        fs.writeFileSync(process.env.MOCHI_RESULT_PATH, `${report}\n`);
    }
    process.stdout.write(`${report}\n`);
    vm.quit();
};

main().catch(error => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
