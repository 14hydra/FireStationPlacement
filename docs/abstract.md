# Abstract

**Title:** Optimizing Fire Station Placement with Street-Aware Travel Times  
**Authors:** Ricky Zhao and Sarp Akalin (ASI)  
**Fair:** CSEF / ISEF

---

Fire engines do not travel at a constant speed: signals, turns, 20 mph streets and junctions change how long a trip takes. Placement studies usually ignore that and treat travel time as a function of distance alone. We ask whether stations chosen with travel times learned from street characteristics reach real London calls faster than stations chosen by distance-based formulas.

We train gradient-boosted decision trees (LightGBM) on London Fire Brigade engine trips to predict driving time from the legal OpenStreetMap route’s street traits, time of day and context—without station IDs, so the model can score sites that do not yet exist. On 200,624 held-out 2025 trips the model’s mean absolute error is **58.9 s**, against **68.7 s** for the classic Kolesar formula on road distance. Interaction features and calibration aimed at outer long drives (from a station embedding analysis) contribute to that gain.

Those predictions feed a placement optimiser (p-median integer program with busy-engine and fairness rules). A learned dispatch model estimates which of the fastest open stations sends the first engine. On London’s real 2014 ten-station closures—trained only on 2009–2013 data—dispatch raises the correlation of predicted vs recorded area slowdowns from **0.48** (assume nearest) to about **0.56**. Recommended layouts that move up to 10 stations with no borough worse than today cut citywide first-engine time by roughly **2.5–5.5 s** on average and improve the slowest 10% of calls, under every travel-time scorer we tested.

Street-aware travel times beat distance formulas on held-out calls and on a historical closure test; using them for placement produces layouts that still win when scored by the older formulas.

---

**Word count:** ~250  

**Keywords:** fire station location; emergency response; travel time; LightGBM; London Fire Brigade; 2014 closures

**One-sentence pitch:** We learn how London streets slow fire engines, then use those times—not raw distance—to place stations, and we check both against real LFB calls and the 2014 closures.
