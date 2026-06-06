/**
 * @desc Auto-version script for @agenteam/providers.
 *
 *   Uses git tags (e.g. v1.0.0, v1.1.0) as the source of truth.
 *   Commits after a tag auto-increment the patch number.
 *
 *   Tag format: /^v\d+\.\d+\.\d+$/  (e.g. v1.0.0, v1.1.0, v2.3.0)
 *
 *   Usage:
 *     node scripts/auto-version.js        # dry-run: print version
 *     node scripts/auto-version.js --apply # write to package.json
 */

import { readFileSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const PKG_PATH = resolve(ROOT, "package.json");

function git(...args) {
  try {
    return execSync(`git ${args.join(" ")}`, { cwd: ROOT, encoding: "utf-8", timeout: 5000 }).trim();
  } catch {
    return null;
  }
}

function buildVersion() {
  const allTags = git("tag", "--sort=-version:refname", "--list", "v*");
  const latestTag = allTags?.split("\n").find(t => /^v\d+\.\d+\.\d+$/.test(t.trim()))?.trim();

  if (!latestTag) {
    const pkg = JSON.parse(readFileSync(PKG_PATH, "utf-8"));
    return pkg.version ?? "0.0.0";
  }

  const base = latestTag.replace(/^v/, "");

  const count = git("rev-list", "--count", `${latestTag}..HEAD`);
  const n = count !== null ? parseInt(count, 10) : 0;
  if (n === 0) return base;

  const [major, minor, patch] = base.split(".").map(Number);
  return `${major}.${minor}.${patch + n}`;
}

function applyVersion(version) {
  const raw = readFileSync(PKG_PATH, "utf-8");
  const pkg = JSON.parse(raw);
  if (pkg.version === version) return false;
  pkg.version = version;
  writeFileSync(PKG_PATH, JSON.stringify(pkg, null, 2) + "\n");
  return true;
}

const version = buildVersion();
const shouldApply = process.argv.includes("--apply");

if (shouldApply) {
  const updated = applyVersion(version);
  console.log(`[auto-version] ${version}${updated ? "" : " (unchanged)"}`);
} else {
  console.log(version);
}
