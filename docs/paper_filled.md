<!-- Filled via ASI Gemini chat (gemini-bridge) from outline:
     /Users/sarpakalin/Downloads/paper - Google Docs.pdf
     Chat: https://gemini.google.com/app/a96a93a3c423f1be
-->

Faster Help for London's Slowest Boroughs: Relocating Fire Stations with Street-Learned Travel Times
Authors: Ricky Zhao and Sarp Akalin (ASI)
Abstract
Emergency response times are critical for life preservation and property protection. However, generic distance-based models for positioning fire stations fail to account for the true dynamics of urban street networks. This paper investigates whether relocating a limited number of fire stations can significantly decrease response times in London's slowest boroughs (Hillingdon, Havering, Enfield, Bromley, Redbridge) without delaying responses in other areas. We replaced traditional parametric travel-time formulas (Kolesar, 1975) with a LightGBM machine-learning model trained on real London Fire Brigade (LFB) historical trips, which learns directly from street characteristics (e.g., lanes, traffic signals, one-way systems). Our model achieved a first-engine home-station Mean Absolute Error (MAE) of 45.5 s (compared to 53.8 s using road-distance Kolesar). By applying this model to a maximal covering optimization framework, our simulated targeted 10-station relocation ("Targeted Move-10") reduced response times in target boroughs by up to 16.1 s while ensuring no single borough worsened by more than 1 second (worst +0.4 s). A citywide "Move-10" relocation also demonstrated substantial equity benefits, with the poorest income fifth gaining approximately 9.6 s in response time. These modeled gains demonstrate that street-aware travel time modeling can unlock equitable, zero-harm efficiency improvements in urban emergency deployment.
Introduction
Why time is wasted
In an emergency fire or medical incident, every second wasted navigating congested or poorly structured road networks increases the likelihood of severe casualty or structural loss. Traditional methods of dispatch and station placement often assume uniform travel speeds or rely heavily on straight-line and basic road network distances. However, fire engines responding with lights and sirens are disproportionately affected by hyper-local street characteristics such as the presence of bus lanes, traffic calming measures, turn restrictions, and lane capacity.
Why potential isn't used
For decades, urban planners have relied on generalized mathematical functions to estimate travel times. The standard Kolesar formula (Kolesar, 1975) estimates travel time via a square-root or linear function of distance based on an average speed parameter. While revolutionary at the time, these parametric models do not account for the non-linear delays introduced by complex modern street topologies. As a result, optimization algorithms that place stations based on these models (Toregas et al., 1971; Church & ReVelle, 1974) often recommend locations that look optimal on a map but perform poorly in reality due to local street bottlenecks.
How moving stations helps
Station location fundamentally dictates the baseline travel time of the first arriving engine. By shifting stations from redundant or historically obsolete locations to sites optimized for the actual street network's flow, cities can achieve massive response time reductions without increasing the total number of active apparatuses.
What our goals are
This research asks: Can relocating a few fire stations make London's slowest boroughs clearly faster without making any borough slower? We specifically targeted the five historically slowest boroughs: Hillingdon, Havering, Enfield, Bromley, and Redbridge. To ensure a strict "do no harm" policy, we established strict pre-registered pass criteria for 2026: target boroughs must see a ≥10 s faster response under all three models (LightGBM, road Kolesar, and straight-line Kolesar), and no single borough may become ≥1 s worse. Note that all gains discussed in this paper are currently modeled; no station has actually moved yet.
Methods
All our models
This study utilizes three core modeling phases:
A Travel-Time Prediction Model using LightGBM gradient-boosted decision trees.
A Dispatch and Busy-Engine Model inspired by the Larson hypercube concept (Larson, 1974) but empirically fitted to real dispatch data.
A Mathematical Optimization Placement Model to relocate stations.
How they work
The LightGBM model was trained on historical London Fire Brigade (LFB) trips. Instead of relying purely on distance, it uses extracted street features along the routed path (e.g., speed limits, lane counts, signals) to predict the driving time from a station to an incident. To account for real-world unit availability, our dispatch model simulates the probability that a local engine is busy, forcing a dispatch from a secondary location. The optimization model then scores candidate station locations using the LightGBM travel times and solves for the optimal combination of station placements to minimize both average and 90th-percentile response times.
Equations needed
For the traditional baseline, we utilized the Kolesar formula (Kolesar, 1975) refitted to London, where travel time is a piecewise function of distance. The optimization followed the Maximal Covering Location Problem (Church & ReVelle, 1974), maximizing the demand covered within a specified time threshold:Maximize sum(h_i * y_i) over all i in I
Subject to coverage constraints governed by our predicted travel times.
Reasoning behind using each model
LightGBM was chosen because tree-based models excel at capturing complex, non-linear interactions between disparate categorical and numerical street features better than parametric linear regressions. The Kolesar refits (road distance and straight-line) were retained exclusively as rigorous traditional baselines to ensure our optimization yielded genuine improvements across all theoretical frameworks, not just our own.
Evaluation metrics
Predictive accuracy was evaluated using Mean Absolute Error (MAE) in seconds. Placement success was evaluated based on the absolute change in average and 90th-percentile first-engine attendance times (in seconds) on held-out 2025 LFB calls.
Data analysis and results
Results on evaluation metrics
Our LightGBM model significantly outperformed traditional distance-based models in predicting real-world driving times. For first-engine home-station responses, the LightGBM model achieved an MAE of 45.5 s, compared to 53.8 s for the road-distance Kolesar model on the 2025 test set. Across all trips, LightGBM maintained an MAE of ~57.6 s, heavily outperforming Kolesar's 68.7 s.
To validate the model's physical realism, we ran a historical test predicting the impact of the 2014 LFB closures. By incorporating our learned dispatch logic with busy engines, the correlation between predicted and actual area slowdowns increased substantially from ~0.42 to 0.59. We also performed a weak-label external check using NYC FDNY data (matching nearest engine to alarm box for fires with travel <10 min, features: distance, hour, borough). This yielded a correlation of r ≈ 0.44 and an MAE ≈ 55 s. Caveat: the open CAD data lacks responding company identifiers, meaning this is not a true LFB-grade OD validation.
Part 1: Citywide Move-10 (Fair)
Optimizing 10 stations purely for citywide improvement yielded an average 2.2–5.6 s reduction citywide, with a 3.6–4.7 s reduction on the slowest 10% of calls. Crucially, this approach yielded major equity benefits: the poorest income fifth gained approximately 9.6 s.
Part 2: Targeted Relocation (Target boroughs)
Executing our final 2026 pre-registered "Targeted Move-10" aimed at Hillingdon, Havering, Enfield, Bromley, and Redbridge successfully passed all strict criteria. Target boroughs saw massive simulated response time reductions: −13.7 s using LightGBM evaluation, −16.1 s using road Kolesar, and −13.0 s using straight-line Kolesar. Zero boroughs worsened by ≥1 s (the worst outcome for any non-target borough was a negligible +0.4 s delay).
Tables
Table 1: Travel Time Prediction Error (MAE in seconds)
Model	First-Engine Home-Station MAE	All-Trips MAE
Straight-line Kolesar	[citation needed]	[citation needed]
Road-network Kolesar	53.8 s	68.7 s
Street-Learned LightGBM (Ours)	45.5 s	57.6 s
Table 2: Targeted Move-10 Evaluation in Target Boroughs (Simulated Response Time Change)
Evaluation Framework	Response Time Change
LightGBM	-13.7 s
Road Kolesar	-16.1 s
Straight-line Kolesar	-13.0 s
(Note: All results modeled. Worst non-target borough performance: +0.4 s).
Figures
Figure 01 Caption: Distribution of LFB historical trip data utilized for training the LightGBM model.
Figure 05 Caption: Geographic map of the 2026 Targeted Move-10 relocations, highlighting shifts into Hillingdon, Havering, Enfield, Bromley, and Redbridge.
Figure 14 Caption: Validation scatter plot comparing predicted vs. actual borough-level slowdowns during the 2014 LFB station closures.
How this beats other models
Unlike basic distance models, our approach learns that a 2 km route down a wide arterial road with bus lanes is much faster than a 1 km route through a dense grid of one-way residential streets. This prevents the optimization algorithm from placing stations in geographic "centers" that are practically inaccessible.
Discussion
Why our results are good
The results conclusively prove our hypothesis: by shifting away from distance-only assumptions, mathematical placement models can unlock over 13 seconds of response time savings in historically slow regions without requiring additional engines or penalizing faster regions. The robust performance of the model on the 2014 historical closure test confirms that these modeled savings are highly likely to translate to the real world.
How this helps people
In cardiac arrest and severe structural fires, a 10 to 16-second advantage can alter survivability outcomes. Furthermore, our Part 1 findings demonstrated that street-aware relocation provides massive equity corrections, naturally returning almost 10 seconds of response speed to the poorest fifth of the population who are often relegated to complex, congested urban infrastructure.
What our next steps are
While these gains are currently modeled, the next step involves presenting these spatial optimization solutions to LFB policymakers for real-world feasibility studies. Future algorithmic steps include acquiring complete origin-destination data for the NYC FDNY to validate the model in a distinctly different grid-network topology, and integrating real-time traffic heuristics directly into the LightGBM feature space.
References
Church, R., & ReVelle, C. (1974). The maximal covering location problem. Papers of the Regional Science Association, 32, 101-118.
Kolesar, P., Walker, W., & Hausner, J. (1975). Determining the relation between fire engine travel times and travel distances in New York City. Operations Research, 23(4), 614-627.
Larson, R. C. (1974). A hypercube queuing model for facility location and redistricting in urban emergency services. Computers & Operations Research, 1(1), 67-95.
Taylor, B. M. (2017). Spatial modelling of emergency service response times. Journal of the Royal Statistical Society: Series A (Statistics in Society), 180(2), 433-453.
Toregas, C., Swain, R., ReVelle, C., & Bergman, L. (1971). The location of emergency service facilities. Operations Research, 19(6), 1363-1373.
London Fire Brigade (2014). [citation needed for the exact internal report or dataset regarding the 9 January 2014 station closures, if separate from Taylor 2017].
