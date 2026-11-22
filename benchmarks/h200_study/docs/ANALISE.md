# Consolidação B07-B14 — otimização extrema da H200

Data: 22/11/2026
Ambiente: H200 NVL do Enterprise NVIDIA H200 NVL Cluster
Modelo: Qwen/Qwen2.5-7B-Instruct
Runtime: vLLM 0.25.1

## Escopo

Foram executadas as baterias planejadas B07 a B14 em uma única janela,
repetindo a infraestrutura e o modelo para separar os efeitos de precisão,
concorrência, prefix cache, chunked prefill, KV cache e estabilidade.

Todos os testes de serving concluíram as requisições sem erro HTTP nos pontos
medidos. A B14 foi encerrada pelo timeout intencional de 1.800 s após 30
minutos; isso não indica falha do servidor.

## B07 — ablação de precisão

A B07 comparou pesos e KV cache separadamente. Nos pontos de concorrência 32 e
128, o resultado foi:

| Pesos | KV | Saída c32 | Saída c128 | Interpretação |
|---|---|---:|---:|---|
| BF16 | BF16 | 4.544,88 tok/s | 9.201,76 tok/s | baseline |
| FP8 | BF16 | 5.159,55 tok/s | 10.190,54 tok/s | ganho principal dos pesos FP8 |
| BF16 | FP8 | 4.580,98 tok/s | 9.780,49 tok/s | KV FP8 sozinho muda pouco o throughput |
| FP8 | FP8 | 5.267,88 tok/s | 10.436,95 tok/s | melhor combinação medida |

[MEDIDO] Em concorrência 32, FP8/FP8 entregou 15,9% mais tokens de saída que
BF16/BF16. Em concorrência 128, o ganho foi 13,4%.

[CONCLUSÃO] O maior ganho de decode veio da redução dos bytes dos pesos. KV
FP8 tem seu valor principal na capacidade de contexto, não no throughput bruto
nesta carga de 7B.

## B08 — curva de concorrência

Configuração FP8/FP8, `max-num-batched-tokens=32768`:

| Concorrência | Saída | TTFT mediano | TTFT P99 | TPOT mediano |
|---:|---:|---:|---:|---:|
| 1 | 243,62 tok/s | 49,92 ms | 52,88 ms | 3,92 ms |
| 2 | 498,23 tok/s | 21,36 ms | 29,71 ms | 3,94 ms |
| 4 | 966,68 tok/s | 25,49 ms | 39,87 ms | 4,05 ms |
| 8 | 1.863,00 tok/s | 28,33 ms | 47,46 ms | 4,20 ms |
| 16 | 3.423,20 tok/s | 29,21 ms | 123,95 ms | 4,56 ms |
| 32 | 5.297,67 tok/s | 32,57 ms | 186,41 ms | 5,85 ms |
| 64 | 8.418,35 tok/s | 42,87 ms | 417,72 ms | 7,15 ms |
| 128 | 10.382,34 tok/s | 122,45 ms | 908,48 ms | 10,81 ms |
| 192 | 10.266,93 tok/s | 535,58 ms | 1.386,42 ms | 14,51 ms |
| 256 | 11.449,47 tok/s | 1.013,83 ms | 1.992,96 ms | 16,23 ms |

[CONCLUSÃO] A vazão continua crescendo até 256, mas o custo de latência cresce
muito depois de 128. Para baixa latência, 16-64 é a faixa mais equilibrada.
Para throughput máximo, 256 é possível, mas não atende um SLO interativo
restritivo.

## B09 — prefix caching

Teste customizado com 64 requisições por padrão, todas respondendo HTTP 200:

| Padrão | Latência média | Mediana | Resultado |
|---|---:|---:|---|
| Prefixo idêntico | 234,56 ms | 242,80 ms | melhor |
| 8 grupos de prefixos | 206,58 ms | 195,25 ms | melhor média do lote |
| Prefixos únicos | 1.664,39 ms | 1.646,75 ms | 7,1x mais lento que idêntico |

[MEDIDO] O prefixo compartilhado reduziu fortemente a latência em relação aos
prefixos únicos. O script não coletou o contador interno de hit rate nessa
bateria; portanto, a conclusão é de latência observada, não de percentual de
hits do cache.

## B10 — chunked prefill

Varredura de `max-num-batched-tokens` entre 2.048 e 65.536.

- Com concorrência 32, a saída subiu de 1.403 tok/s em 2.048 para
  1.863 tok/s em 65.536, enquanto o TTFT mediano subiu de 268 ms para 1.844 ms.
- Com concorrência 128, a saída ficou aproximadamente entre 4.318 e 4.465
  tok/s, mas o TTFT mediano variou de 355 ms a 2.946 ms.
- O TPOT tende a cair com valores maiores, mas o primeiro token espera mais.

[CONCLUSÃO] `max-num-batched-tokens=32768` não é um valor universalmente ótimo.
Para interação, valores menores controlam o atraso do primeiro token. Para
throughput de lote, valores maiores são aceitáveis.

## B11 — KV cache em contexto longo

Comparação FP8 nos pesos com KV BF16 versus KV FP8:

| Carga | KV BF16 | KV FP8 | Diferença de saída |
|---|---:|---:|---:|
| 8.192 entrada / 256 saída | 967,81 tok/s | 978,97 tok/s | +1,15% |
| 16.384 entrada / 128 saída | 237,67 tok/s | 238,99 tok/s | +0,56% |

TTFT, TPOT e P99 ficaram praticamente iguais. [CONCLUSÃO] Nesta carga, KV FP8
não trouxe ganho de throughput relevante; seu benefício esperado é comportar
mais tokens e sessões antes de esgotar a VRAM.

## B12 — max-num-seqs

| max-num-seqs | Saída | TTFT mediano | TTFT P99 | TPOT mediano |
|---:|---:|---:|---:|---:|
| 32 | 3.027,92 tok/s | 10.141,73 ms | 20.150,08 ms | 7,37 ms |
| 64 | 3.899,22 tok/s | 7.629,01 ms | 15.061,41 ms | 11,66 ms |
| 128 | 4.305,10 tok/s | 6.294,21 ms | 12.660,64 ms | 21,94 ms |
| 256 | 4.507,43 tok/s | 6.195,83 ms | 10.233,36 ms | 32,38 ms |
| 512 | 4.497,20 tok/s | 6.220,47 ms | 10.270,82 ms | 32,40 ms |

[CONCLUSÃO] Aumentar `max-num-seqs` acima de 256 não trouxe ganho: 512 ficou
ligeiramente abaixo de 256. O teto prático desta carga é aproximadamente 256.

## B13 — qualidade determinística FP8 versus BF16

Foram usados cinco prompts idênticos, temperatura 0 e 96 tokens máximos.

- Exact match: 0 de 5.
- Similaridade textual média: 0,0958.
- Saída BF16 média: 506,4 caracteres.
- Saída FP8 média: 413,4 caracteres.
- Um prompt FP8 produziu apenas 15 caracteres.

[MEDIDO] As respostas divergiram fortemente.

[CONCLUSÃO] O resultado não autoriza declarar FP8 equivalente a BF16. A causa
mais provável é a limitação já observada no checkpoint: q_scale, k_scale e
prob_scale da atenção ficaram em 1.0, sem calibração. FP8 deve permanecer fora
de produção para qualidade até novo teste com checkpoint/escalas calibradas.

## B14 — estabilidade

- Execução limitada por `timeout 1800`, código 124 esperado.
- O servidor permaneceu funcional até o fim da janela.
- Não houve falha HTTP registrada nos benchmarks coletados.
- O teste de estabilidade usado foi uma carga sintética constante; não é uma
  prova de estabilidade de produção com distribuição real de prompts.

## Recomendações objetivas

1. Para throughput máximo: FP8/FP8, `max-num-seqs=256`, concorrência alta.
2. Para serviço interativo: começar com concorrência 16-64 e medir SLO; não
   usar automaticamente 256.
3. Manter prefix caching para workloads com prefixos repetidos.
4. Ajustar `max-num-batched-tokens` conforme o SLO: menor para TTFT, maior para
   lote/throughput.
5. Não promover FP8 para produção sem escalas de atenção calibradas e uma
   avaliação de qualidade adequada.
6. Repetir B09 capturando métricas internas do vLLM (`prefix_cache_queries` e
   `prefix_cache_hits`) para transformar a inferência de latência em evidência
   direta.

Arquivos brutos: `resultados/work-results/` e `execucao/run-final.log`.
