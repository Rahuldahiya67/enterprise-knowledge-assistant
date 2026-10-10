# Evaluation report

Generator: `offline-extractive` · Judge: `token-overlap`

| Metric | Value |
|---|---|
| cases | 30 |
| hit_rate@5 | 1.0 |
| recall@5 | 1.0 |
| mrr | 1.0 |
| answer_accuracy | 0.962 |
| faithfulness | 1.0 |
| hallucination_rate | 0.0 |
| refusal_accuracy | 0.967 |

## Cases

| id | answerable | hit | facts | faithful | refusal ok |
|---|---|---|---|---|---|
| hr-01 | True | True | 1.00 | 1.00 | True |
| hr-02 | True | True | 1.00 | 1.00 | True |
| hr-03 | True | True | 1.00 | 1.00 | True |
| hr-04 | True | True | 1.00 | 1.00 | True |
| hr-05 | True | True | 1.00 | 1.00 | True |
| hr-06 | True | True | 1.00 | 1.00 | True |
| hr-07 | True | True | 1.00 | 1.00 | True |
| hr-08 | True | True | 1.00 | 1.00 | True |
| hr-09 | True | True | 1.00 | 1.00 | True |
| fin-01 | True | True | 1.00 | 1.00 | True |
| fin-02 | True | True | 1.00 | 1.00 | True |
| fin-03 | True | True | 1.00 | 1.00 | True |
| fin-04 | True | True | 1.00 | 1.00 | True |
| fin-05 | True | True | 1.00 | 1.00 | True |
| fin-06 | True | True | 1.00 | 1.00 | True |
| fin-07 | True | True | 1.00 | 1.00 | True |
| it-01 | True | True | 0.00 | - | False |
| it-02 | True | True | 1.00 | 1.00 | True |
| it-03 | True | True | 1.00 | 1.00 | True |
| it-04 | True | True | 1.00 | 1.00 | True |
| ops-01 | True | True | 1.00 | 1.00 | True |
| ops-02 | True | True | 1.00 | 1.00 | True |
| ops-03 | True | True | 1.00 | 1.00 | True |
| qa-01 | True | True | 1.00 | 1.00 | True |
| qa-02 | True | True | 1.00 | 1.00 | True |
| filter-01 | True | True | 1.00 | 1.00 | True |
| neg-01 | False | - | - | - | True |
| neg-02 | False | - | - | - | True |
| neg-03 | False | - | - | - | True |
| neg-04 | False | - | - | - | True |

## Misses to look at

**it-01**: What is the minimum password length?

> I couldn't find this in the available documents.

