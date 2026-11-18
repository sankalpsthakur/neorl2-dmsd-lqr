# Provenance: order of commits in the private working repository (times in UTC)

This kit is an export without history. The list below shows that the pre-registration preceded the code, that Amendment 1 preceded every run, and that the frozen configuration (Amendment 2) and the final script's hash were committed before the simulator was contacted. Dates are the committer's clock; there is no external timestamp.

```
a47d95d | 2026-11-18 15:01:34 | neorl2-dmsd: BARS.md and PREREGISTRATION.md (before any code, kernel or score)
8bc1265 | 2026-11-18 15:04:23 | neorl2-dmsd: numpy replica of the DMSD simulator and metric, identification, LQR design, validation-only selection, claim rule, tests
bb4271c | 2026-11-18 15:08:53 | neorl2-dmsd: stage logic, four CPU kernels (generated, statically checked, not pushed), scripts, ledgers, STATUS, TEST-LOG
0389051 | 2026-11-18 16:13:50 | neorl2-dmsd: PREREGISTRATION Amendment 1 (honest provenance, data used, hard-coded claim completeness, G0c report-only, regime check, caps); BARS wording
4a12818 | 2026-11-18 16:14:19 | neorl2-dmsd: hard-coded claim completeness, in-kernel wall caps, slot release, G0c report-only, regime check, tests (34), kernels regenerated
1f1a19e | 2026-11-18 16:14:37 | neorl2-dmsd: TEST-LOG announced rows and script binding, kernel versions
0935d50 | 2026-11-18 16:15:03 | neorl2-dmsd: STATUS runbook and finding-by-finding
faa2b92 | 2026-11-18 16:25:39 | neorl2-dmsd: dry run outcome recorded (official neorl2 imports on Kaggle, G0b exact, internet on)
a8bdb99 | 2026-11-18 16:32:39 | neorl2-dmsd: Amendment 2, frozen configuration c3f031e99dbdfa12 (prep passed; before g0 and final)
036f9b5 | 2026-11-18 16:32:48 | neorl2-dmsd: final kernel bound to the frozen sha and script sha
51ff90a | 2026-11-18 16:45:26 | neorl2-dmsd: final run recorded (claim rule met: 97.78 against EDAC 78.65; lead re-score matches); results, test log, ledger
```

Left out of the export: the working status file, raw kernel logs, and verbatim copies of five NeoRL2 source files kept privately to pin what was read (their hashes are in `protocol/sources.sha256`; the kernels fetch them from the NeoRL2 repository at commit b50d91d0).
