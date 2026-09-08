#!/usr/bin/env node
// Run from bindings/node before publishing. Syncs the version committed in
// each npm/*/package.json and the root package's optionalDependencies block
// to match the root package.json's own version, so a version bump doesn't
// also require hand-editing five other files to stay consistent.
"use strict";

const fs = require("fs");
const path = require("path");

const root = JSON.parse(fs.readFileSync("package.json", "utf8"));
const version = root.version;

for (const dep of Object.keys(root.optionalDependencies || {})) {
  root.optionalDependencies[dep] = version;
}
fs.writeFileSync("package.json", JSON.stringify(root, null, 2) + "\n");

for (const dir of fs.readdirSync("npm")) {
  const pkgPath = path.join("npm", dir, "package.json");
  const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
  pkg.version = version;
  fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");
}

console.log(`synced all package versions to ${version}`);
