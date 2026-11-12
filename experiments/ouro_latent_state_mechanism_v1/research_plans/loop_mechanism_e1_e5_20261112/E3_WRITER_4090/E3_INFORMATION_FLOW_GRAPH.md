# E3 information-flow evidence graph

```mermaid
flowchart LR
  Q["Q input: question + choices + formatting"] -->|"same-question Q9 substitution; actual intervention"| R["R attention updates L9–L16"]
  R -->|"full-bundle add/remove; Train and Val confirmed"| F["final candidate competition"]
  R -->|"E=L9 fails; L=L10–L16 qualifies and passes nested confirmation"| L["distributed L candidate"]
  R -. "L18 state distances: diagnostic" .-> X["L18 R state / attention"]
  R -->|"parent full bundle plus controlled KQ-to-R edge"| K["L18 key-to-R edge"]
  K -->|"same receiver causal edge intervention; posthoc factorial summary"| F
  L -. "specific L-only mediation edge not yet independently cut" .-> K
  U["earlier input/loop computation"] -. "information origin unresolved" .-> Q
```

Solid labeled intervention arrows concern their fixed receiver and actual manipulated variables; they are not a complete natural causal route. Dashed arrows indicate diagnostics or unresolved mechanisms. The Q region is not a semantic-only question-stem source. The full-bundle edge factorial does not by itself identify a mediator of L-only effects; existing E4/E5 studies use other operators and do not validate that new path.
