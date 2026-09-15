# Published flow check — failed live-run acceptance

Tested the unchanged submitted URL, https://top-kernel-demo.andre520395.chatgpt.site/?demo=1, in the browser on September 13, 2026.

Input: `https://github.com/rasbt/LLMs-from-scratch`, optimization **Both**, dataset **TinyStories**. This replaced an initial minGPT test at the user's request to avoid Karpathy repositories.

| Step | Observed result |
| --- | --- |
| Submit repository | Accepted the URL; displayed “Recorded demo.” |
| Dataset selection | Displayed fixed Tiny Shakespeare and TinyStories choices, including nanoGPT-specific wording for the unrelated repository. |
| Start | Displayed “Restoring the recorded evaluations…”; no live run ID was created by this flow. |
| Diagram | Animated generations and working/kept/failed states. The Repo link pointed to `flappybird1084/top-k`, despite `requested_repo=rasbt/LLMs-from-scratch` in the URL. |
| Console | Automatically opened after the replay. Displayed historical architecture data and FineWeb-Edu, despite choosing TinyStories. |
| Notebook | Embedded export rendered. It explicitly said “Static notebook” and “not connected to a kernel.” |
| Final-results button | Worked. Displayed historical modern-lm architecture (7.21%) and separate 21.6M kernel workload (3.30%), not measurements of the submitted repository. |

Source confirmation in the deployed source checkout: `dist/front.js` bypasses `/api/runs` when `demo=1`, calling `replayIntake()` and `finishReplayData()`. `dist/assets/final.js` selects `window.RECORDED_RUNS`. The deployment injects `demo=1` into entry pages. Thus this is a working recorded presentation, not a successful live training test.

Live acceptance remains blocked until the production gateway and GPU worker path are deployed to this same Site. Retest must verify a real run ID, repository-specific dataset discovery, fresh GPU evaluations, matching repository/data identity throughout, live observability, and final metrics derived from that run. Do not relabel the historical metrics as results of the submitted repository.

## Live bridge deployment

Published source `75d098378cda2299f9ea76bcc940983e010a6542` on the same submitted URL. Removed forced replay routing and historical metric fallbacks. Browser submission created run `43f2ee89aaac59ec0547dd2c4915e9ac` for Raschka's repository and entered real OAuth repository discovery. API ownership checks return 404 for another browser's run and for admin routes.

The initial API smoke run cloned the repository, authored and verified an adapter, loaded real data, and reached GPU profiling. It failed because the restricted process environment lacked USER/LOGNAME; both are now set for subsequent runs. Its failed status is preserved. A fresh W&B run was created under top-k-judges.

One notebook returned 410 Gone. Both modes now queue serially on the remaining notebook until another GPU is supplied. Credentials remain outside Git and are not passed to repository processes. The Unix-identity isolation is not a VM boundary. The judging gateway expires at 2026-09-14T02:16:01Z, with a five-minute heartbeat watching the service. Full optimization completion is not yet verified.
