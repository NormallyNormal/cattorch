#!/usr/bin/env node
/* Run a generated benchmark in the official Scratch VM under Node.

Usage:
  node benchmarks/run_scratch_vm.cjs benchmarks/artifacts/cattorch_suite.sb3 [--inspect-output] [--inspect-lists] [--list-values-prefix=PREFIX]

The package is intentionally not a cattorch runtime dependency. Install the
official VM in a separate Node environment and expose its node_modules through
NODE_PATH when necessary.
*/

'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const VirtualMachine = require('@scratch/scratch-vm');

const RESULTS_NAME = 'cattorch benchmark results';

const main = async () => {
    const projectPath = process.argv[2];
    const inspectOutput = process.argv.includes('--inspect-output');
    const inspectLists = process.argv.includes('--inspect-lists');
    const valuesPrefixArgument = process.argv.find(
        argument => argument.startsWith('--list-values-prefix=')
    );
    const valuesPrefix = valuesPrefixArgument
        ? valuesPrefixArgument.slice('--list-values-prefix='.length)
        : null;
    if (!projectPath) {
        throw new Error('Usage: run_scratch_vm.cjs <benchmark.sb3>');
    }

    const vm = new VirtualMachine();
    vm.setTurboMode(true);
    await vm.loadProject(fs.readFileSync(projectPath));
    vm.start();

    const wallStart = process.hrtime.bigint();
    const completion = new Promise((resolve, reject) => {
        const timeout = setTimeout(
            () => reject(new Error('Scratch VM benchmark timed out after 15 minutes')),
            15 * 60 * 1000
        );
        vm.once('PROJECT_RUN_STOP', () => {
            clearTimeout(timeout);
            resolve();
        });
    });
    vm.greenFlag();
    await completion;
    const wallSeconds = Number(process.hrtime.bigint() - wallStart) / 1e9;

    const stage = vm.runtime.getTargetForStage();
    const resultVariable = Object.values(stage.variables).find(
        variable => variable.name === RESULTS_NAME
    );
    if (!resultVariable) {
        throw new Error(`Project has no stage list named ${RESULTS_NAME}`);
    }

    const report = {
        project: path.resolve(projectPath),
        turbo_mode: vm.runtime.turboMode,
        wall_seconds: wallSeconds,
        results: resultVariable.value.map(String)
    };
    if (inspectOutput) {
        report.outputs = vm.runtime.targets
            .filter(target => !target.isStage)
            .map(target => {
                const output = Object.values(target.variables).find(
                    variable => variable.name === 'output' && Array.isArray(variable.value)
                );
                if (!output) return null;
                let argmax = -1;
                let maximum = -Infinity;
                output.value.forEach((raw, index) => {
                    const value = Number(raw);
                    if (value > maximum) {
                        maximum = value;
                        argmax = index;
                    }
                });
                return {
                    target: target.getName(),
                    length: output.value.length,
                    argmax,
                    maximum
                };
            })
            .filter(Boolean);
    }
    if (inspectLists) {
        const interesting = /(?:output|result|values|ids)$/;
        report.lists = vm.runtime.targets
            .flatMap(target => Object.values(target.variables)
                .filter(variable =>
                    Array.isArray(variable.value) &&
                    interesting.test(variable.name) &&
                    variable.name !== RESULTS_NAME
                )
                .map(variable => {
                    const numeric = variable.value.map(Number);
                    return {
                        target: target.getName(),
                        name: variable.name,
                        length: variable.value.length,
                        numeric_sum: numeric.reduce((sum, value) => sum + value, 0),
                        sha256: crypto.createHash('sha256')
                            .update(JSON.stringify(variable.value))
                            .digest('hex'),
                        first: variable.value.slice(0, 5),
                        last: variable.value.slice(-5)
                    };
                })
            );
    }
    if (valuesPrefix !== null) {
        report.list_values = vm.runtime.targets
            .flatMap(target => Object.values(target.variables)
                .filter(variable =>
                    Array.isArray(variable.value) &&
                    variable.name.startsWith(valuesPrefix)
                )
                .map(variable => ({
                    target: target.getName(),
                    name: variable.name,
                    values: variable.value.map(Number)
                }))
            );
    }
    console.log(JSON.stringify(report, null, 2));
    vm.quit();
};

main().catch(error => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
