# Abstract

**Title:** Faster Help for London's Slowest Boroughs: Relocating Fire Stations with Street-Learned Travel Times
**Authors:** Ricky Zhao and Sarp Akalin (ASI)
**Fair:** Synopsys / CSEF / ISEF

---

In 2024 the first fire engine took 45 s longer to reach calls in London's five slowest boroughs than the city average
(371 s against 326 s), and four of the five missed the Brigade's 6-minute average target. Can relocating a few stations
make these boroughs clearly faster without making any borough slower?

A LightGBM model trained on 2021-2024 London Fire Brigade engine trips predicts driving time from the street route (2025
error 45.5 s against 53.8 s for the standard formula). A placement model allows for busy engines. Before searching, we
fixed the targets (the five boroughs with the slowest recorded 2024 times) and the pass marks: at least 10 s faster
under LightGBM and two distance formulas, and no borough 1 s or more slower.

On held-out 2025 calls, 15 relocations sped up the target boroughs by 14.1-17.2 s, passing both marks; 10 relocations
missed by 0.2 s under one model. A later search limited to sites near commercial or industrial land passed with 10
relocations (10.8-14.3 s). A citywide search gives these boroughs about half as much. The model's 18 s outer- London
under-prediction is constant across trip lengths, so it cancels between layouts.

A prediction fixed before using January-July 2026 calls passed in full: 13.0-16.1 s faster, no borough slower. The gain
is concentrated: 25-36 s in Havering and Redbridge, bringing Havering within the 6-minute target under every model.
Small, targeted relocations can give the worst-served areas a clear gain with every borough protected.

---

**Word count:** 248 (ISEF limit 250)

**Keywords:** fire station location; emergency response equity; travel time; LightGBM; London Fire Brigade

**One-sentence pitch:** We learn how London's streets slow fire engines, then move a few stations so the boroughs that wait
longest get 13-16 s faster help on average (25-36 s in Havering and Redbridge) while no borough gets slower.

---

## Earlier abstract (Part 1 only, kept for the record)

Part 1 asked whether stations chosen with street-learned travel times reach real London calls faster than stations chosen
by distance formulas. Its abstract (git history before 2026-11-05) reported a 58.9 s model error on all 2025 trips against
68.7 s for the Kolesar formula, a 2014-closure correlation of about 0.56 with dispatch, and Move 10 layouts cutting citywide
attendance by about 2.5-5.5 s. Later results changed some of these: the error that applies to placement is 45.5 s on
first-engine trips (results.md 1.1b), the dispatch correlation is 0.59 (3.2), and the neutral test (6.1) found that the
learned model picks nearly the same sites as the formulas (under 1 s difference). Part 1's answer is therefore: the learned
model predicts times better, but choosing with it barely changes the layout.
