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
    classDef diagram_neutral fill:#FBFCF8,stroke:#D5DECE,color:#35492F,stroke-width:1px
    classDef diagram_working fill:#EDF4F7,stroke:#A8C6D5,color:#3C647A,stroke-width:1px
    classDef diagram_kept fill:#EDF5E7,stroke:#A8C694,color:#416738,stroke-width:1px
    classDef diagram_retry fill:#F8EEE7,stroke:#D3AD91,color:#8C6247,stroke-width:1px
    classDef diagram_winner fill:#2D4938,stroke:#2D4938,color:#FFFFFF,stroke-width:2px
    class R,A,B diagram_neutral
    class O,C,V diagram_working
    class K,P,E diagram_kept
    class F diagram_retry
    class W diagram_winner
    linkStyle default stroke:#94A48B,stroke-width:1.5px`;
mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',
  themeVariables:{fontFamily:'DM Sans, Arial, sans-serif',fontSize:'14px',background:'#fcfdf8',lineColor:'#94A48B',edgeLabelBackground:'#fcfdf8'},
  flowchart:{htmlLabels:false,curve:'basis',nodeSpacing:45,rankSpacing:28,padding:14}});
mermaid.render('pitch-search-mermaid',pitchFlowSource).then(({svg})=>{
  document.querySelector('#pitch-mermaid').innerHTML=svg;
}).catch(()=>{
  document.querySelector('#pitch-mermaid').textContent='The diagram could not load. Refresh to try again.';
});
