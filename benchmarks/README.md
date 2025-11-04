# Benchmarks

This document contains detailed benchmarks for all nanoBEIR datasets, and some additional benchmarks with batch sizes.

## Full nanoBEIR results

Here are the full nanoBEIR results for each teacher/NIFE pair.

### `NIFE-mxbai-embed-large-v1`

| dataset            |   ndcg@10 teacher |   ndcg@10 NIFE |
|:-------------------|------------------:|---------------:|
| NanoArguAna        |              0.67 |           0.64 |
| NanoClimateFEVER   |              0.40 |           0.32 |
| NanoDBPedia        |              0.64 |           0.61 |
| NanoFEVER          |              0.92 |           0.87 |
| NanoFiQA2018       |              0.56 |           0.44 |
| NanoHotpotQA       |              0.87 |           0.75 |
| NanoMSMARCO        |              0.66 |           0.62 |
| NanoNFCorpus       |              0.38 |           0.34 |
| NanoNQ             |              0.71 |           0.60 |
| NanoQuoraRetrieval |              0.96 |           0.90 |
| NanoSCIDOCS        |              0.45 |           0.38 |
| NanoSciFact        |              0.79 |           0.74 |
| NanoTouche2020     |              0.53 |           0.47 |

The NIFE model performs worse on most datasets, but performs closely to its teacher on DBPedia and Arguana. Note that the NIFE model was trained on MSMARCO, so this represents a case of in-domain results: if you have many documents and can train on them, your NIFE model can approach the performance of your teacher model.

### `NIFE-gte-modernbert-base`

| dataset            |   ndcg@10 teacher |   ndcg@10 NIFE |
|:-------------------|------------------:|---------------:|
| NanoArguAna        |              0.77 |           0.61 |
| NanoClimateFEVER   |              0.46 |           0.38 |
| NanoDBPedia        |              0.60 |           0.61 |
| NanoFEVER          |              0.94 |           0.81 |
| NanoFiQA2018       |              0.62 |           0.54 |
| NanoHotpotQA       |              0.77 |           0.65 |
| NanoMSMARCO        |              0.65 |           0.64 |
| NanoNFCorpus       |              0.35 |           0.35 |
| NanoNQ             |              0.72 |           0.63 |
| NanoQuoraRetrieval |              0.97 |           0.90 |
| NanoSCIDOCS        |              0.47 |           0.37 |
| NanoSciFact        |              0.82 |           0.75 |
| NanoTouche2020     |              0.48 |           0.46 |

As you can see the NIFE model performs worse on most datasets, but outperforms the base model on DBPedia and is pretty close on NanoMSMARCO. Note that the NIFE model was trained on MSMARCO, so this represents a case of in-domain results: if you have many documents and can train on them, your NIFE model can approach the performance of your teacher model.

## Effect of batch size on speed

One interesting issue is that most query embedders are not run on very large batch sizes. In real-life workloads, embedders are often run in microservices that ingest a single query at a time. Therefore, reporting QPS at high batch sizes is actually not a realistic estimate of performance.

For this experiment, I use `NIFE-mxbai-embed-large-v1`, all timings are done on CPU (Macbook Pro M3)

| Batch size     |   NIFE QPS |   Teacher QPS |   x Speedup |
|----:|-------:|----------:|------------:|
|   1 |  12925 |        13 |        1007 |
|   2 |  14090 |        26 |         552 |
|   4 |  19670 |        45 |         435 |
|   8 |  31326 |        64 |         486 |
|  16 |  46346 |        82 |         567 |
|  32 |  62077 |        99 |         628 |
|  64 |  75661 |       107 |         706 |
| 128 |  93948 |       101 |         931 |

As you can see both NIFE and the teacher benefit a lot from batching, although NIFE is already very fast to begin with. This shows that, on CPU, there is a very large gain on QPS by switching from the teacher to NIFE.
