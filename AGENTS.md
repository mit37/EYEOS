# STANDARDS.md — rules every rebuild follows

> Put this file in the root of **every** repo next to `PRD.md`. Also copy it as `AGENTS.md` (Codex and several other agents read that name automatically) and `CLAUDE.md` (Claude Code reads that name). Whatever agent builds the repo must read this file and `PRD.md` before writing any code.
> If `PRD.md` and this file disagree, this file wins.

---

## 1. Honesty rules (these protect Mitansh's reputation. They are not optional.)

1. **This is a v2 rebuild, and the repo says so.** The first line under the title in every README is:
   > **v2 (2026 rebuild).** The original prototype's code was not preserved; this is a clean re-implementation of the same design. Numbers in this README come from this codebase.
2. **Never backdate commits.** No `GIT_AUTHOR_DATE` / `GIT_COMMITTER_DATE` tricks and no `--date` flag. Commit history shows the real build dates.
3. **Never copy a number from mitanshm.com into a README.** Every metric in a README (accuracy, test count, latency, P(cascade), tick cost, etc.) must be produced by a script in this repo and be reproducible with one command. If a number can't be measured in the cloud instance, say "not measured" and explain why.
4. **Negative results get reported.** If an experiment or benchmark comes out worse than expected, report it in the README's Results section. Never delete it.
5. **State limitations up front.** Every README has a "What this does not do" section (see Warrant's README for the tone).
6. **Credit the tools.** The README footer says: "Built with AI coding agents (<the agent(s) actually used, e.g. Claude Code, OpenAI Codex, Google Antigravity, Freebuff/GLM>) under my direction; design, specs, review and evaluation are mine." Name the real agent. It's true, and it reads better than hiding it.

## 2. Secrets and safety

- No API keys, tokens, `.env` files, or personal data are ever committed. Add `.gitignore` entries for `.env*`, `*.key`, `*.pem`, `secrets/`, model weights (`*.gguf`, `*.safetensors`, `*.bin`, `*.onnx`, `*.tflite`, `*.task`), datasets, and recordings.
- Ship a `.env.example` with placeholder names only.
- Add **gitleaks** to CI (`gitleaks/gitleaks-action@v2`) and run `gitleaks detect` locally before the first push.
- Any feature that calls a paid API needs a **spend cap** (hard limit on tokens or requests per run) and a **dry-run / mock mode** that works with no key. CI always runs in mock mode.
- Demos must run with **no API key** when that's feasible (recorded fixtures or local models).
- Never commit model weights or third-party datasets. Provide `scripts/download_*.sh`, which checks the license and pulls them at build time.

## 3. Repo conventions

| Item | Rule |
|---|---|
| Owner | `github.com/mit37/<repo-name>` (kebab-case, name given in each PRD) |
| License | MIT unless the PRD says otherwise. Respect upstream licenses (AGPL/MPL forks keep their license). |
| Default branch | `main`, protected once CI is green |
| Commits | Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), small and frequent, one logical change each |
| CI | GitHub Actions: lint → typecheck → test → build → gitleaks. Badge in README. |
| Formatting | Language-standard formatter enforced in CI (Prettier/ESLint, ruff+black, rustfmt+clippy, ktlint, StyLua+selene, gdformat) |
| Tests | Every "acceptance criterion" in the PRD maps to at least one automated test. Test count is shown in README (generated, not typed). |
| Releases | Tag `v2.0.0` when the Definition of Done is met; attach build artifacts where applicable (APK, desktop bundle, web build). |
| Topics | Add 5–8 GitHub topics per repo (listed in each PRD). |
| Description | One-line repo description, given in each PRD. |

## 4. README template (every repo)

```
# <Name>
> v2 (2026 rebuild) line (see §1.1)

<one-sentence pitch, same as the repo description>

[CI badge] [license badge] [release badge]

<demo GIF or 60–90s video link, above the fold>

## The thirty-second version        (problem → what it does → why it's interesting)
## What's in it                     (capability table)
## Results                          (generated numbers + the command that produced them)
## Architecture                     (ASCII or Mermaid diagram)
## Running it                       (≤5 commands, no key needed for the demo)
## Testing                          (how to run, what's covered, generated test count)
## What this does not do            (honest limitations)
## Design decisions                 (3–6 bullets: choice → why → trade-off)
## Credits & licenses
```

## 5. Demo assets

- Record a GIF (≤8 MB, via `vhs`, `asciinema` + `agg`, or Playwright video → `ffmpeg`) for CLI/web tools. For GUI/mobile, leave a `docs/DEMO.md` script so Mitansh can record it himself in 5 minutes.
- If the project can run in a browser (web builds, Godot web export), deploy to **GitHub Pages** and link it at the top of the README.

## 6. How the coding agent should work through a PRD

1. Read `STANDARDS.md` then `PRD.md`. Restate the Definition of Done in `docs/PLAN.md`.
2. Scaffold, CI and a failing smoke test first. Get CI green before building features.
3. Build milestones **in order**. At the end of each milestone: all tests green, commit, update `docs/PLAN.md` checkboxes.
4. When a PRD requirement is impossible in the cloud instance (no GPU, webcam, phone or macOS), build it behind an interface, test it with a mock/fake, and record it under "Cloud-instance constraints" in `docs/PLAN.md` and in the README limitations.
5. Don't expand scope. Anything marked **Non-goal** stays out, even if it would be easy.
6. Finish with the Definition of Done checklist, all items ticked, then tag `v2.0.0`.

## 7. After each repo ships (Mitansh's part)

- Update the project's page on mitanshm.com so its claims match the v2 README (new numbers, "v2 rebuild" note, repo link).
- Pin the best 6 repos on the GitHub profile.
- Record any demo that needs real hardware (phone, webcam, drone sim) and add it to the README.
