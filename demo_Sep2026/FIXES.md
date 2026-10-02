# Fixes to deploy after the demo

1. Map PC names to microscope names (e.g. `THORLABS_30_016` → Musashi / Shinano).
   `darkcurrent/registry.json` already records `THORLABS_30_016` as Shinano.
2. Figure out how to update and use the catalog best, including which scan
   settings matter for matching and which do not. Note: ChanA dark-current
   entry 3 (row 6) is used by v4 on only 5 of 80 sampled frames of its own
   movie, because single frames are too noisy on that row.
3. Decide how cleaning experience is added to the PMT catalog continuously
   (today only v2.2 writes entries; v3/v4 never do). Ideally the app has a
   button that sends the collected entries from a colleague's runs back to the
   repo, e.g. as a pull request, so the shared catalog grows from every run.
4. Re-evaluation of prioritized seeds in the catalogue (A / B / C order:
   same-day DarkCurrent or stored shutter, then older DC/shutter, then
   `live_clean`). Decide whether previous successful cleans should rank
   higher, and whether one geometry per stack is the right catalog unit.
5. Add catalogue-writing to the v4 version for shipping. v4 must be able to
   store geometry from a finished run (today only v2.2 appends `live_clean`).

## Questions to address

1. Why are the diagonals the veto for linescan congruence? (H/V give \(q\);
   a seed is kept only if that \(q\) predicts the measured main and anti
   diagonal periods.)
2. Predicting tilt and spatial frequency from congruent line-scans in
   real-space. Consider how to make best prediction and ensure that the
   center vs. edge problem is not conflating the measurements.
