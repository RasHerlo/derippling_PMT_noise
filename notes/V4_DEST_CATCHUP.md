# v4 dest batch — catch-up list

Updated: 2026-10-01 12:15Z (after full 414-job walk, run `20260928T161433Z`)

The ECF1 dest walk **finished**: 414/414 logged. Last stack
`M1_RV` SSC LED only `…LED_ex03` ChanB completed (99.9% active, median RMS 4.16).

This resume: **ok=56**, **skipped=357**, **err=1**.

Earlier `Permission denied` dests that **caught up** on this resume:

- job 276 ChanB `M1_RLV` PFC LED only `…LED500micW_min10_ex02` (99.8%, RMS 2.69)
- job 318 ChanB `M1_RLV` SSC 240927 LED only `…ex01` (98.4%, RMS 2.46)
- job 319 ChanA `M1_RLV` SSC 240927 LED only `…ex02` (72.2%, RMS 1.8)

## Still open (retry)

None. Job 368 ChanB completed 2026-10-01 (run `20261001T151613Z`): 97.9% active, median RMS 3.54. All 414 ECF1 dest stacks have a successful v4 dest folder.

Catchup-list refresh hit `WinError 1392` (corrupted/unreadable) on
`F:\CollectedData\M1_RLV_pdf\New recordings SSC 240927\LED+AP\240927_pl100_pc240_LED+AP25microW_ex01\DATA\ChanB\defringe_v4`
— filesystem, not the algorithm. Worth a disk check later.
