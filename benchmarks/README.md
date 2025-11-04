# Benchmarks

This document contains detailed benchmarks for all nanoBEIR datasets, and some additional benchmarks with batch sizes.

## Full nanoBEIR results

Here are the full nanoBEIR results for each teacher/NIFE pair.

### `NIFE-mxbai-embed-large-v1`

| dataset            |   ndcg@10 teacher |   ndcg@10 NIFE |
|:-------------------|------------:|------------:|
| NanoArguAna        |    0.668786 |    0.640516 |
| NanoClimateFEVER   |    0.396524 |    0.318884 |
| NanoDBPedia        |    0.636832 |    0.612894 |
| NanoFEVER          |    0.922807 |    0.874241 |
| NanoFiQA2018       |    0.562228 |    0.438247 |
| NanoHotpotQA       |    0.872424 |    0.75016  |
| NanoMSMARCO        |    0.660706 |    0.620861 |
| NanoNFCorpus       |    0.38455  |    0.344202 |
| NanoNQ             |    0.707486 |    0.600752 |
| NanoQuoraRetrieval |    0.955517 |    0.89957  |
| NanoSCIDOCS        |    0.450356 |    0.381717 |
| NanoSciFact        |    0.788312 |    0.738594 |
| NanoTouche2020     |    0.531303 |    0.471938 |

The NIFE model performs worse on most datasets, but performs closely to its teacher on DBPedia and Arguana. Note that the NIFE model was trained on MSMARCO, so this represents a case of in-domain results: if you have many documents and can train on them, your NIFE model can approach the performance of your teacher model.

### `NIFE-gte-modernbert-base`

| dataset            |   ndcg@10 teacher |   ndcg@10 NIFE |
|:-------------------|------------:|------------:|
| NanoArguAna        |    0.770357 |    0.612167 |
| NanoClimateFEVER   |    0.460153 |    0.376965 |
| NanoDBPedia        |    0.599664 |    0.60707  |
| NanoFEVER          |    0.942215 |    0.813864 |
| NanoFiQA2018       |    0.619847 |    0.541698 |
| NanoHotpotQA       |    0.77488  |    0.645495 |
| NanoMSMARCO        |    0.647244 |    0.641866 |
| NanoNFCorpus       |    0.352195 |    0.346539 |
| NanoNQ             |    0.721093 |    0.634101 |
| NanoQuoraRetrieval |    0.96713  |    0.897534 |
| NanoSCIDOCS        |    0.473989 |    0.374358 |
| NanoSciFact        |    0.820234 |    0.75267  |
| NanoTouche2020     |    0.475784 |    0.460761 |

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
