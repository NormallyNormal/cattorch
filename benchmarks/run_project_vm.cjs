#!/usr/bin/env node
/* Run a project from cattorch.build_project in the official Scratch VM.

Usage:
  node benchmarks/run_project_vm.cjs project.sb3 [--answer=TEXT ...] [--key=space]
      [--list=NAME=JSON ...] [--var=NAME=VALUE ...] [--no-flag]
      [--broadcast=MESSAGE ...] [--out=FILE]

Sets each --list (a JSON array) and --var given before the first --broadcast
on the stage, then clicks the green flag (unless --no-flag) or presses --key,
and waits for the project to stop. The remaining options run in order: a
--broadcast is sent and waited for, as "broadcast and wait" does, and a --list
or --var is set, so each call can get its own inputs. Each "ask and wait"
gets the next --answer (an empty answer once they run out). Prints the
stage's global lists and variables plus everything sprites said, as JSON, or
writes it to --out, since the VM logs warnings to stdout.
*/

'use strict';

const fs = require('fs');
const VirtualMachine = require('@scratch/scratch-vm');

const values = prefix => process.argv
    .filter(argument => argument.startsWith(prefix))
    .map(argument => argument.slice(prefix.length));

const assignment = text => {
    const split = text.indexOf('=');
    return [text.slice(0, split), text.slice(split + 1)];
};

const main = async () => {
    const projectPath = process.argv[2];
    const answers = values('--answer=');
    const key = values('--key=')[0];
    const output = values('--out=')[0];
    const vm = new VirtualMachine();
    vm.setTurboMode(true);
    await vm.loadProject(fs.readFileSync(projectPath));
    vm.start();
    const stage = vm.runtime.getTargetForStage();
    const named = name => {
        const variable = Object.values(stage.variables).find(entry => entry.name === name);
        if (!variable) throw new Error(`no stage variable or list named ${JSON.stringify(name)}`);
        return variable;
    };
    // Options in command-line order; sets before the first broadcast
    // happen before the green flag.
    const steps = process.argv.slice(3).flatMap(argument => {
        for (const kind of ['--list=', '--var=', '--broadcast=']) {
            if (argument.startsWith(kind)) return [[kind, argument.slice(kind.length)]];
        }
        return [];
    });
    const set = (kind, text) => {
        const [name, value] = assignment(text);
        named(name).value = kind === '--list=' ? JSON.parse(value) : value;
    };
    while (steps.length && steps[0][0] !== '--broadcast=') set(...steps.shift());

    const said = [];
    const questions = [];
    vm.runtime.on('SAY', (target, type, text) => said.push({sprite: target.sprite.name, text}));
    vm.runtime.on('QUESTION', question => {
        if (question === null) return;
        questions.push(question);
        setTimeout(() => vm.runtime.emit('ANSWER', answers.length ? answers.shift() : ''), 0);
    });
    const stopped = () => new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('project timed out')), 10 * 60 * 1000);
        vm.runtime.once('PROJECT_RUN_STOP', () => {
            clearTimeout(timeout);
            resolve();
        });
    });
    if (key) {
        const done = stopped();
        vm.postIOData('keyboard', {key: key.replace('space', ' '), isDown: true});
        await done;
    } else if (!process.argv.includes('--no-flag')) {
        const done = stopped();
        vm.greenFlag();
        await done;
    }
    for (const [kind, message] of steps) {
        if (kind !== '--broadcast=') {
            set(kind, message);
            continue;
        }
        const done = stopped();
        const threads = vm.runtime.startHats('event_whenbroadcastreceived', {
            BROADCAST_OPTION: message,
        });
        if (!threads || threads.length === 0) throw new Error(`nothing receives ${message}`);
        await done;
    }
    const data = {};
    for (const variable of Object.values(stage.variables)) {
        if (variable.type !== 'broadcast_msg') data[variable.name] = variable.value;
    }
    const result = JSON.stringify({questions, said, stage: data}, null, 1);
    if (output) {
        fs.writeFileSync(output, result);
    } else {
        console.log(result);
    }
    vm.quit();
};

main().catch(error => {
    console.error(error);
    process.exit(1);
});
