/* Local Mermaid bundle keeps the pitch usable offline and from file://. */
'use strict';
const pitchFlowSource = `flowchart TD
    R["Repository<br/>Training code + dataset"]
    A["Adapter agent<br/>Model · Data · Loss"]
    B["Baseline<br/>Default recipe · 5 min"]
    O["Orchestrator<br/>Strategy + research"]
    C["8 subagents<br/>Author in parallel"]
    V["Validation<br/>1-min screening · matched baseline"]
    F["Repair + retry"]
    K["Keep top K<br/>Rank measured results"]
    P["Next generation<br/>Selected parents + curator lessons"]
    E["Final retrain<br/>Equal 5-min budgets"]
    W["Verified winner"]
    R --> A --> B --> O
    O --> C --> V
    C -->|Build fails| F
    V -->|Checks fail| F
    F --> C
    V -->|Valid results| K
    K -->|Budget remains| P
    P --> O
    K -->|Search complete| E --> W
    classDef diagram_neutral fill:#19191F,stroke:#42424D,color:#EDEDF0,stroke-width:1px
    classDef diagram_working fill:#1A1E35,stroke:#8B97FF,color:#C4CAFF,stroke-width:1px
    classDef diagram_kept fill:#14251F,stroke:#345F4D,color:#8DCEB0,stroke-width:1px
    classDef diagram_retry fill:#2B2217,stroke:#80643F,color:#F2B560,stroke-width:1px
    classDef diagram_winner fill:#2D4938,stroke:#2D4938,color:#FFFFFF,stroke-width:2px
    class R,A,B diagram_neutral
    class O,C,V diagram_working
    class K,P,E diagram_kept
    class F diagram_retry
    class W diagram_winner
    linkStyle default stroke:#94A48B,stroke-width:1.5px`;
mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',
  themeVariables:{fontFamily:'DM Sans, Arial, sans-serif',fontSize:'14px',background:'#111114',lineColor:'#94A48B',edgeLabelBackground:'#111114',textColor:'#ededf0'},
  flowchart:{htmlLabels:false,curve:'basis',nodeSpacing:45,rankSpacing:28,padding:14}});
mermaid.render('pitch-search-mermaid',pitchFlowSource).then(({svg})=>{
  document.querySelector('#pitch-mermaid').innerHTML=svg;
}).catch(()=>{
  document.querySelector('#pitch-mermaid').textContent='The diagram could not load. Refresh to try again.';
});
