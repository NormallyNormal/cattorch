#!/usr/bin/env node
/* Run a project from cattorch.build_project in the official Scratch VM.

Usage:
  node benchmarks/run_project_vm.cjs project.sb3 [--answer=TEXT ...] [--key=space]

Clicks the green flag (or presses --key), answers each "ask and wait" with the
next --answer (an empty answer once they run out), and prints the stage's
global lists and variables plus everything sprites said, as JSON.
*/

'use strict';

const fs = require('fs');
const VirtualMachine = require('@scratch/scratch-vm');

const main = async () => {
    const projectPath = process.argv[2];
    const answers = process.argv
        .filter(argument => argument.startsWith('--answer='))
        .map(argument => argument.slice('--answer='.length));
    const key = process.argv.find(argument => argument.startsWith('--key='));
    const vm = new VirtualMachine();
    vm.setTurboMode(true);
    await vm.loadProject(fs.readFileSync(projectPath));
    vm.start();

    const said = [];
    const questions = [];
    vm.runtime.on('SAY', (target, type, text) => said.push({sprite: target.sprite.name, text}));
    vm.runtime.on('QUESTION', question => {
        if (question === null) return;
        questions.push(question);
        setTimeout(() => vm.runtime.emit('ANSWER', answers.length ? answers.shift() : ''), 0);
    });
    const done = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('project timed out')), 10 * 60 * 1000);
        vm.runtime.once('PROJECT_RUN_STOP', () => {
            clearTimeout(timeout);
            resolve();
        });
    });
    if (key) {
        vm.postIOData('keyboard', {key: key.slice('--key='.length).replace('space', ' '), isDown: true});
    } else {
        vm.greenFlag();
    }
    await done;
    const stage = vm.runtime.getTargetForStage();
    const data = {};
    for (const variable of Object.values(stage.variables)) {
        if (variable.type !== 'broadcast_msg') data[variable.name] = variable.value;
    }
    console.log(JSON.stringify({questions, said, stage: data}, null, 1));
    vm.quit();
};

main().catch(error => {
    console.error(error);
    process.exit(1);
});
