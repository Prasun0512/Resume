const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../assets/js/profile-assistant.js'), 'utf8');
const context = { document: { addEventListener() {} }, window: {}, navigator: {} };
// Expose the pure router without starting the browser widget or downloading a model.
vm.runInNewContext(source.replace('document.addEventListener("DOMContentLoaded", initAssistant);',
  'globalThis.answer = routeQuestion; globalThis.profile = profile;'), context);
assert.match(context.answer('Where does Prasun work now at IBM?'), /3 November 2026/);
assert.equal(context.profile.experience.filter(x => x.dates.includes('Present')).length, 1);
assert.match(context.answer('Tell me about ML Release Governance'), /independent portfolio/i);
assert.match(context.answer('Tell me about AI Delivery Planner'), /optimization/);
assert.match(context.answer('Summarize Prasun'), /IBM/);
console.log('5 profile consistency checks passed');
