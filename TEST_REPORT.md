# PATH-Mat Research Console verification report

Date: 2026-09-03

## Automated tests

- 13 backend/model tests passed.
- 6 Playwright browser tests passed.
- 19 automated tests passed in total.
- Real EC-PathWM model load passed (`scikit-learn 1.9.0`).
- All 12 configured agent contracts loaded.
- Full CentralBrain chain emitted all 10 ordered stages with event hashes.
- Missing-source abstention passed.
- Unseen-source model abstention passed.
- Physical-actuation blocking passed.
- Invalid-formula controlled error handling passed.
- JSON and CSV upload tests passed.
- Report generation and download path passed.
- Regression reproduced and fixed: isotonic calibration flattened all nearby
  condition outputs to the same route-level value (52.6882%); the experiment
  page had incorrectly repeated this value for every temperature/time pair.
- The experiment page now uses EC-PathWM's pre-calibration condition signal for
  within-batch ranking. The default 18-condition grid produces nine distinct
  displayed scores (25.51–25.82), while the calibrated 52.7% remains only on
  the route-level overview.
- Feature-support negative control passed: temperature and duration coefficients
  are active; the deployed artifact has no atmosphere-category coefficient.
  Air/inert pairs therefore retain equal scores and the interface explains that
  atmosphere is used for branch coverage rather than inventing an effect.
- The training feature builder now emits `atmosphere=<category>` features so a
  future retrained artifact can learn atmosphere effects from suitable records.
- Public Gradio endpoint was verified online.
- Dynamic recommendation branches were verified for the default competing-path
  case, missing-XRD case, physical-policy block and unseen-source abstention.
- Plain-language result checks passed: the page states that the two possible
  results are close, explains what the percentages compare, and provides a
  staged validation plan.
- Public-language audit passed: report-style phrasing and unexplained labels
  such as “反方审查” and “科学门控” are absent from the main interface.
- Desktop browser main flow passed.
- Mobile 390×844 layout passed with no horizontal overflow.
- The model-guided experiment page passed on desktop and 390×844 mobile,
  including the responsive workload chart and condition table.
- The light result card, input panel and CSV download label passed explicit
  dark-theme text-color checks; the white-on-pale rendering regression is fixed.
- The generated first-round CSV contains all 18 candidate conditions, the four
  selected conditions and blank fields for sample IDs and measured results.
- Both Plotly/WebGL 3D pages rendered successfully; the route landscape passed
  a drag-rotation interaction check and the Agent network canvas was verified.
- A time-bounded Cloudflare public session returned HTTP 200 and its public
  Gradio API returned the model-guided experiment summary, all 18 condition
  rows and the responsive workload figure.

## Current public verification

- URL: `https://caroline-boats-history-runtime.trycloudflare.com/`
- Verified: 2026-09-03 21:50 (Asia/Shanghai)
- Scheduled expiry: 2026-09-04 21:45:13 (Asia/Shanghai)
- Public page: HTTP 200.
- Public `/screen_experiments` API: 4 outputs returned; 18 ranked condition
  rows present; nine distinct condition scores present; temperature-change
  negative control passed; workload-reduction summary present; generated CSV
  download verified and readable.

## Scientific boundary

The application exposes retrospective public-record proxy outputs. Physical
actuation and production simulation remain disabled. The tests verify software
behavior and policy enforcement, not experimental materials performance.
