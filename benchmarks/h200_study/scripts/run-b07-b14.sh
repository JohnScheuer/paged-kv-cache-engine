#!/usr/bin/env bash
set -euo pipefail

MODEL='Qwen/Qwen2.5-7B-Instruct'
ROOT=${RESULT_ROOT:-/cache/results}
rm -rf "$ROOT"
mkdir -p "$ROOT" /work/inductor /work/triton
export PATH=/usr/local/bin:$PATH
export HF_HOME=/cache/hf HF_HUB_OFFLINE=1 HOME=/work USER=bench
export TORCHINDUCTOR_CACHE_DIR=/work/inductor TRITON_CACHE_DIR=/work/triton VLLM_NO_USAGE_STATS=1

log(){ printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$ROOT/master.log"; }

start_server(){
  local name=$1 quant=$2 kv=$3 max_tokens=$4 max_seqs=$5
  local slog="$ROOT/${name}-server.log"
  local extra=()
  [[ "$quant" != none ]] && extra+=(--quantization "$quant")
  extra+=(--kv-cache-dtype "$kv")
  log "START $name quant=$quant kv=$kv batched=$max_tokens seqs=$max_seqs"
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    "${extra[@]}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens "$max_tokens" \
    --max-num-seqs "$max_seqs" \
    --gpu-memory-utilization 0.92 \
    --port 8000 --host 127.0.0.1 >"$slog" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null; then
      log "READY $name seconds=$((i*5)) pid=$SERVER_PID"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "FAILED_START $name"
      cat "$slog" >> "$ROOT/master.log"
      return 1
    fi
    sleep 5
  done
  log "TIMEOUT_START $name"
  return 1
}

stop_server(){
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=''
  sleep 3
}

run_bench(){
  local name=$1 concurrency=$2 prompts=$3 input=$4 output=$5 prefix=$6
  log "BENCH $name concurrency=$concurrency prompts=$prompts input=$input output=$output prefix=$prefix"
  timeout 900 vllm bench serve --backend vllm --model "$MODEL" --dataset-name random \
    --random-input-len "$input" --random-output-len "$output" \
    --random-prefix-len "$prefix" --num-prompts "$prompts" \
    --max-concurrency "$concurrency" --port 8000 --temperature 0 \
    >"$ROOT/${name}.log" 2>&1 || log "BENCH_EXIT name=$name code=$?"
  grep -E 'Successful requests|Benchmark duration|Request throughput|Output token throughput|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median ITL|P99 ITL|Prefix cache' "$ROOT/${name}.log" >> "$ROOT/summary.log" || true
}

python_requests(){
  local name=$1 mode=$2
  python3 - "$ROOT" "$name" "$mode" <<'PY'
import concurrent.futures, json, sys, time, requests
root, name, mode = sys.argv[1:]
url='http://127.0.0.1:8000/v1/completions'
base=('You are evaluating a production inference engine. Explain the relationship between memory bandwidth, '
      'KV cache management, continuous batching and decode latency on a Hopper GPU. ' * 80)
if mode == 'same-prefix':
    prompts=[base + f'\nRequest-specific question {i}: give one concise sentence.' for i in range(64)]
elif mode == 'unique-prefix':
    prompts=[(base.replace('Hopper GPU', f'Hopper GPU variant {i}') + f'\nQuestion {i}: give one concise sentence.') for i in range(64)]
elif mode == 'groups':
    prompts=[]
    for g in range(8):
        p=base + f'\nShared group {g}. '
        prompts.extend([p + f'Question {i}: give one concise sentence.' for i in range(8)])
else:
    prompts=[base + f'\nWarm request {i}.' for i in range(32)]

def one(item):
    i,p=item; t=time.perf_counter()
    try:
        r=requests.post(url,json={'model':'Qwen/Qwen2.5-7B-Instruct','prompt':p,'max_tokens':32,'temperature':0},timeout=300)
        return {'i':i,'status':r.status_code,'latency_ms':(time.perf_counter()-t)*1000}
    except Exception as e:
        return {'i':i,'status':0,'error':repr(e),'latency_ms':(time.perf_counter()-t)*1000}
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
    out=list(ex.map(one,enumerate(prompts)))
json.dump({'mode':mode,'results':out},open(f'{root}/{name}.json','w'),indent=2)
PY
}

run_cfg(){
  local name=$1 quant=$2 kv=$3 max_tokens=$4 max_seqs=$5
  start_server "$name" "$quant" "$kv" "$max_tokens" "$max_seqs"
  run_bench "${name}-c1" 1 256 2048 256 1536
  run_bench "${name}-c32" 32 256 2048 256 1536
  run_bench "${name}-c128" 128 256 2048 256 1536
  run_bench "${name}-c256" 256 512 2048 256 1536
  stop_server
}

: > "$ROOT/summary.log"
echo 'B07 A/B PRECISION' | tee -a "$ROOT/master.log"
run_cfg b07-bf16-bf16 none bfloat16 32768 256
run_cfg b07-fp8-bf16 fp8 bfloat16 32768 256
run_cfg b07-bf16-fp8 none fp8 32768 256
run_cfg b07-fp8-fp8 fp8 fp8 32768 256

echo 'B08 CONCURRENCY' | tee -a "$ROOT/master.log"
start_server b08-sweep fp8 fp8 32768 256
for c in 1 2 4 8 16 32 64 128 192 256; do run_bench "b08-c${c}" "$c" 256 2048 256 1536; done
stop_server

echo 'B09 PREFIX CACHE' | tee -a "$ROOT/master.log"
start_server b09-prefix fp8 fp8 32768 256
python_requests b09-same-prefix same-prefix
python_requests b09-unique-prefix unique-prefix
python_requests b09-groups groups
sleep 20
stop_server

# B10: chunked prefill sweep, one server per batched-token value.
for t in 2048 4096 8192 16384 32768 65536; do
  start_server "b10-t${t}" fp8 fp8 "$t" 256
  run_bench "b10-t${t}-c32" 32 256 4096 128 0
  run_bench "b10-t${t}-c128" 128 256 2048 256 1536
  stop_server
done

# B11: long-context KV ablation.
for kv in bfloat16 fp8; do
  start_server "b11-${kv}" fp8 "$kv" 32768 256
  run_bench "b11-${kv}-long" 32 128 8192 256 4096
  run_bench "b11-${kv}-xlong" 128 128 16384 128 8192
  stop_server
done

# B12: server-side max-num-seqs sweep.
for s in 32 64 128 256 512; do
  start_server "b12-s${s}" fp8 fp8 32768 "$s"
  run_bench "b12-s${s}" 256 256 2048 256 1536
  stop_server
done

# B13: deterministic output comparison. The output files are evidence; no
# quality claim is made until a reviewer inspects them.
for mode in bf16 fp8; do
  if [[ "$mode" == bf16 ]]; then start_server b13-bf16 none bfloat16 32768 128; else start_server b13-fp8 fp8 fp8 32768 128; fi
  python3 - "$ROOT" "$mode" <<'PY'
import json,sys,time,requests
root,mode=sys.argv[1:]
prompts=[
 'Explain in exactly three sentences why LLM decode is often memory-bound.',
 'List four mechanisms used by vLLM to improve serving throughput.',
 'What is the difference between prefill and decode?',
 'Explain why prefix caching reduces repeated prompt work.',
 'Describe one risk of using uncalibrated FP8 scales.'
]
out=[]
for p in prompts:
 t=time.perf_counter(); r=requests.post('http://127.0.0.1:8000/v1/completions',json={'model':'Qwen/Qwen2.5-7B-Instruct','prompt':p,'max_tokens':96,'temperature':0},timeout=300)
 j=r.json(); out.append({'prompt':p,'status':r.status_code,'latency_ms':(time.perf_counter()-t)*1000,'text':j.get('choices',[{}])[0].get('text','')})
json.dump(out,open(f'{root}/b13-{mode}-outputs.json','w'),ensure_ascii=False,indent=2)
PY
  stop_server
done
python3 - "$ROOT" <<'PY'
import json, difflib, pathlib, sys
r=pathlib.Path(sys.argv[1])
a=json.load(open(r/'b13-bf16-outputs.json')); b=json.load(open(r/'b13-fp8-outputs.json'))
out=[]
for x,y in zip(a,b):
 out.append({'prompt':x['prompt'],'exact_match':x['text']==y['text'],'bf16_len':len(x['text']),'fp8_len':len(y['text']),'similarity':difflib.SequenceMatcher(None,x['text'],y['text']).ratio()})
json.dump(out,open(r/'b13-comparison.json','w'),indent=2)
PY

# B14: 30-minute mixed operational load.
start_server b14-stability fp8 fp8 32768 256
(timeout 1800 vllm bench serve --backend vllm --model "$MODEL" --dataset-name random \
 --random-input-len 2048 --random-output-len 256 --random-prefix-len 1536 \
 --num-prompts 20000 --max-concurrency 128 --request-rate 8 --port 8000 --temperature 0 \
 >"$ROOT/b14-30min.log" 2>&1) || log "B14_EXIT code=$?"
stop_server

echo '=== COMPLETE ===' | tee -a "$ROOT/master.log"
# Keep the completed container alive long enough for kubectl cp from the
# persistent result directory; the result files already survive on the PVC.
sleep 900
