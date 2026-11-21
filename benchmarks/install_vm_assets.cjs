#!/usr/bin/env node
/* Copy the stylesheet the scratch-vm Node build loads but doesn't publish.

@scratch/scratch-vm's bundled jsdom reads browser/default-stylesheet.css next
to the bundle when it starts, and the npm package leaves that file out. It is
jsdom's own stylesheet, so this copies it from the installed jsdom. Runs as
the postinstall step of `npm ci` here.
*/

'use strict';

const fs = require('fs');
const path = require('path');

const modules = path.join(__dirname, 'node_modules');
const source = path.join(modules, 'jsdom', 'lib', 'jsdom', 'browser', 'default-stylesheet.css');
const target = path.join(modules, '@scratch', 'scratch-vm', 'browser', 'default-stylesheet.css');
fs.mkdirSync(path.dirname(target), {recursive: true});
fs.copyFileSync(source, target);
