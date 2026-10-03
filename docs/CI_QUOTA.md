# Hosted CI admission and compute budget

Effective 2026-10-03, full scientific validation is requested explicitly. Automatic
quick syntax/lint and documentation checks are not evidence of full validation.
The scientific commands, assertions, environments, Python matrices, shard coverage
verification, and timeouts are unchanged in the new full-validation workflows.

## Request validation

On the PI's Mac (or ask the coding agent to run this command):

```sh
python3 /Users/rossma/.local/share/actions-quota-controller/controller.py request n2n PR_NUMBER
python3 /Users/rossma/.local/share/actions-quota-controller/controller.py request nmb PR_NUMBER
```

Use `--suite atomic` for N2N's atomic-nuclear portable regression, `--now` to
bypass the quiet period, or `--retry` to deliberately retest an already attempted
SHA. None bypasses the budget. Requests expire after 24 hours and must be renewed
explicitly; there is no surprise backlog when the month changes.

```sh
python3 /Users/rossma/.local/share/actions-quota-controller/controller.py status
```

The external launchd controller polls every five minutes while the Mac is awake
and the user session is active. A request waits until its observed PR head has
remained unchanged for 30 minutes (up to another polling interval). A new commit
restarts the quiet period. Draft, closed, and fork PRs are not admitted. Only one
full suite across the two repositories is admitted at a time. The controller
checks the live head again before dispatch; admission rechecks it on GitHub.
Superseded work is cancelled on the next controller tick.

A completed request is consumed: later pushes do not automatically request
another full run. Request again for the next merge candidate. Do not use the
Actions Run workflow button or Re-run jobs as an alternative admission path.
These paths cannot pass admission without the current ticket; workflow reruns
are rejected. The admission job rejects expired or replayed dispatch tickets.

## Budget

| Policy | Minutes |
| --- | ---: |
| Combined full-validation ceiling per UTC calendar month | 1,200 |
| Combined full-validation ceiling per rolling 24 hours | 300 |
| Stop quick CI at combined repository usage per UTC month | 1,800 |

These are conservative operating limits, not a claim about the GitHub account's
plan or billing-cycle dates. No paid allowance was increased. The 1,800-minute
limit leaves 200 minutes relative to a 2,000-minute envelope for other repositories
and in-flight quick work. Other private repositories are not controlled here;
this is not an account-wide GitHub billing cap.

Usage is reconstructed from allocated job durations, rounded per job, across all
attempts in both repositories. Failed jobs that never got a runner count as zero.
The controller includes historical October use; it does not reset usage when
installed. Consequently October is already over budget: full admission is refused
and automatic quick/documentation jobs are switched off before runner allocation.
They can resume after the policy month resets and accounting succeeds.

Before dispatch, reserve the full timeout envelope plus slack: N2N 190 minutes,
NuclearMatter 230, atomic-nuclear 40. Reconcile to measured job usage after
completion. In-flight jobs reserve conservative headroom. Unknown dispatch
outcomes retain the reservation and are never automatically retried. API errors
prevent full admission. The quick-CI guard is periodic, not an instantaneous
hard cap; a network outage can prevent its variable from being updated. Every
quick job also has a short explicit timeout.

The controller fails closed if API pagination would be incomplete, a runner type
is unknown, or accounting cannot be read. Check `status` and the launchd logs
before changing policy. Never delete the state to clear a budget block.

## Workflows and merge evidence

- Old `test.yml`, `atomic-nuclear.yml`, `docs.yml`, and backend `ci.yml` IDs remain
  disabled repository-wide, including historical PR branches. Their default
  branch files are non-running tombstones.
- `quick-ci.yml` runs on PR updates and pushes to main, with cancellation of
  superseded runs. Backend syntax checking does not install the physics stack.
- `docs-ci.yml` restricts push to main and cancels superseded documentation builds.
- `full-validation.yml` and `full-atomic-validation.yml` are dispatch-only. The
  definition is loaded from main; checkout uses the exact admitted candidate SHA
  and does not persist credentials. Test jobs have read-only permissions.
- Admission is gated by the controller's `CI_ADMISSION` repository variable.
  Quick jobs use `CI_QUICK_ENABLED`. Treat these as operational controls.
- The controller publishes `quota/full-full` or `quota/full-atomic` against the
  tested candidate SHA. A green workflow with skipped admission is not accepted.
  Full evidence must be green for the exact merge candidate; quick green alone
  does not satisfy this requirement. Neither repository currently has branch
  protection enforcing this policy, so reviewers/agents must check it.
- Existing PR branches need to incorporate the workflow changes to gain the new
  automatic quick checks. The old expensive workflows stay disabled even before
  that update. Explicit full requests can already test older candidate SHAs.

Keep frequent scientific iteration on existing local/cluster environments. Hosted
full validation is reserved for selected merge candidates. No self-hosted runner
was registered and no scientific assertions or coverage were weakened.

## Operations and verification

Canonical controller source: `scripts/ci_control/controller.py` in the Nucleons2Nucleosynthesis repository.
Installed runtime/state: `/Users/rossma/.local/share/actions-quota-controller/`.
LaunchAgent: `com.rossma.actions-quota-controller` (five-minute interval).
This operates independently of the existing RIFT review coordinator.

Run controller tests with:

```sh
# In the Nucleons2Nucleosynthesis checkout:
python3 -m unittest discover -s scripts/ci_control -p 'test_*.py'
```

Workflow syntax is checked with actionlint. The only suppressed diagnostic is
its constant-false warning on the intentionally non-running retired tombstones.
No full scientific CI was dispatched just to test quota controls.
