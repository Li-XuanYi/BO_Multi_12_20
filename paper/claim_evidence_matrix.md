# Claim--Evidence Matrix

| Claim | Evidence available | Allowed wording | Wording excluded |
|---|---|---|---|
| Objective data source | Simulator databases and run summaries | All reported objectives are simulated; only simulator outputs update the database. | Physical or laboratory validation |
| Chen2020 main result | Five paired seeds: LLMBO-MO `0.3835 +/- 0.0079`, ParEGO `0.3853 +/- 0.0094`; 2/5 wins | Similar observed mean with smaller sample deviation | Mean improvement, superiority, equivalence |
| Ecker2015 main result | Five paired seeds: `1.8684 +/- 0.0027` versus `1.5866 +/- 0.0130`; 5/5 wins | Higher observed mean in all five archived pairs | Cell-independent transfer or population-level superiority |
| Main mechanism | Ten summaries: zero accepted/effective mean lifts; 16 `force_pool` influence events per run | Screened warm start plus early preference-conditioned candidate search under plain EI | Gain caused by posterior-mean lift |
| Same-batch lift execution | Twelve active lifts per Region/Full run; lifted/plain choices differ 3--8 times; historical acquisition path matching telemetry | Region and Full actively use the posterior-covariance mean during the first 12 BO iterations | Diagnostic-only or zero-intervention description |
| Same-batch outcome | Plain `0.3836`, Warm `0.3902`, Region `0.3862`, Full `0.3900`; five seeds | Warm has the highest observed mean; Full has the smallest sample deviation; descriptive component comparison | Significant component benefit, causal generalization, variance reduction |
| Objective preprocessing | New Chen2020 matched runs: min--max `0.4146 +/- 0.0067`, z-score `0.4122 +/- 0.0078`, none `0.3826 +/- 0.0358`; both scaled modes win 4/5 pairs versus none | Min--max and z-score have close observed means; both are higher in four pairs and none is more variable in this batch | Statistical significance, equivalence, or a general claim that one scaling is always best |
| Mean-shift budget | Fixed-payload replay over five seeds: means span `0.4180`--`0.4205`; best observed mean at `B_mu=0.05`; nonmonotonic | Descriptive sensitivity with no monotonic trend | Universal optimum, tuning gain, or significance |
| Degradation quantity | Implemented empirical proxy; no calibration against capacity measurements | Uncalibrated protocol-level degradation proxy in arbitrary units | Capacity fade (%), measured capacity loss, mechanistic degradation state |
| Posterior uncertainty | Implementation changes the acquisition mean and leaves covariance unchanged | Covariance unchanged; posterior-mode bound applies | Reduced uncertainty because of LLM guidance |
| LLM scores | Parsed model-reported scores used for ranking/guidance | LLM-reported heuristic score | Confidence probability or calibrated belief |
| Runtime | Archived Chen2020 timing, but no hardware/software metadata | Within-archive runtime ratio | Portable performance benchmark |
| Pareto/profile figures | Separate tuned GPT-4.1-mini Chen2020 seed-8409 archive | Illustrative archive; full-space nondominated set; low-temperature point is not final nondominated | Evidence for the matched DeepSeek five-seed comparison |
| Laboratory platform | Exact source photograph; no platform measurements in databases | Platform available for future replay | Experimental validation or physical safety evidence |

All means and standard deviations are descriptive sample summaries over five
seeds. No inferential significance or practical-equivalence test is reported.
