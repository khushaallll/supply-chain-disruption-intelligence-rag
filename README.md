# Multi-Hop Agentic RAG for Supply Chain Disruption Monitoring in the Semiconductor Industry

Disruptions like natural disasters or industrial accidents at a single supplier in a supply chain 
network can have catastrophic downstream exposure for several consumer companies, with no 
direct visibility into that exposure. The information to foresee these disruptions is available 
weeks in advance through the media, but the consuming companies only discover their financial 
damage when shipments are delayed. Current monitoring relies on manual analyst review, 
keyword-based alerts, or single-step retrieval from very few sources. It not only fails to surface 
the consequences of a disruption in time but also hinders tracing supplier dependencies at 
multiple levels, causing the supply chain to fail. Moreover, the disclosure lag between the 
occurrence of a disruption and the affected company's own regulatory acknowledgment of it has 
been observed anecdotally but not measured systematically at scale. 

This study proposes a six-layer multi-hop agentic RAG pipeline that monitors the world for 
supply-chain disruption in the semiconductor sector and identifies and quantifies supply-chain 
risk for downstream companies. A structured supplier knowledge graph is constructed from SEC 
EDGAR filings and public trade datasets. It is paired with a document store that contains 
historical incident reports (past disruption news, analyst reports) and supplier-specific documents 
(10-K filings, earnings transcripts, press releases). Disruption events are monitored through 
GDELT and filtered for supply chain relevance using a fine-tuned DeBERTa classifier. A 
ReAct-based LLM agent iteratively traverses the supplier graphs and retrieves evidence from the 
document store, which is synthesised into structured, citation-grounded risk alerts. Generated 
risk alerts are verified for factual faithfulness using NLI-based entailment checking. The system 
is evaluated on 50 historical disruption events from 2020–2024, comparing the proposed system 
against two baselines: keyword-only retrieval and single-hop graph lookup. 

This work is expected to demonstrate that multi-hop agentic reasoning outperforms baselines on 
recall and precision, particularly recall in cases of indirect downstream exposure not mentioned 
explicitly in the news event. Additionally, this study contributes one of the first systematic 
measurements of the disclosure lag between a public news event of a disruption and its 
acknowledgment by the affected company — hypothesised to span from weeks to months and 
vary by event type and supply concentration. These findings are expected to demonstrate the 
advantage of multi-hop reasoning over heterogeneous sources in high-stakes, domain-specific 
retrieval tasks, and to indicate that publicly available information is sufficient to surface supply 
chain risk ahead of formal corporate disclosure.
 
