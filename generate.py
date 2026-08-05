import os
import asyncio
import logging
import argparse
from openai import AsyncOpenAI, RateLimitError

# Setup logging
logging.basicConfig(
    filename='eval_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)
# Suppress noisy logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

CONCURRENCY_LIMIT = 50  # Limit concurrent requests to reduce API throttling.

MODELS = [
    # Boyue models
    "gpt-5.2",
    "gemini-3-pro-preview",
    "claude-sonnet-4-5-20250929",
    "grok-4",
    "gpt-oss-120b",
    "meta-llama/llama-4-maverick",
    # Bailian models
    "qwen3-max",
    "glm-4.7",
    "Moonshot-Kimi-K2-Instruct",
    "qwen3-235b-a22b-instruct-2507",
    "deepseek-v3.2"
]

PROMPTS = {
    "FunctionEC": "Predict the enzymatic function of the given sequence. First, detail your analysis and the evidence supporting your findings. Then, strictly at the end of your response, return one or more Enzyme Commission (EC) numbers in x.x.x.x format.",
    "Modification": "Identify the RNA modifications in the sequence. First, detail your analysis and the evidence supporting your findings. Then, strictly at the end of your response, return the modification names from the list ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm6A', 'm6Am', 'm7G', 'Psi', 'AtoI'] or 'none'.",
    "NoncodingRNAFamily": "Classify the RNA sequence into its specific family. First, detail your analysis and the evidence supporting your findings. Then, strictly at the end of your response, return exactly one of the 13 class names: '5S_rRNA', '5_8S_rRNA', 'tRNA', 'ribozyme', 'CD-box', 'miRNA', 'Intron_gpI', 'Intron_gpII', 'HACA-box', 'riboswitch', 'IRES', 'leader', or 'scaRNA'.",
    "cpd": "Determine if the sequence contains a core promoter. First, detail your analysis and the evidence supporting your findings. Then, strictly at the end of your response, return 'yes' or 'no'.",
    "emp": "Classify whether the given DNA sequence is associated with epigenetic marks based on its sequence patterns. First, briefly explain the sequence-based evidence supporting your finding. Then, make the best-supported positive-or-negative prediction and strictly at the end of your response return 'yes' or 'no'.",
    "pd": "Determine if the sequence acts as a promoter. First, detail your analysis and the evidence supporting your findings. Then, strictly at the end of your response, return 'yes' or 'no'.",
    "tf_h": "Determine if the sequence is a binding site for human transcription factors. First, detail your analysis and the evidence supporting your findings. Then, strictly at the end of your response, return 'yes' or 'no'.",
    "tf_m": "Determine if the sequence is a binding site for mouse transcription factors. First, detail your analysis and the evidence supporting your findings. Then, strictly at the end of your response, return 'yes' or 'no'."
}

# resolve_api removed

def infer_task_from_path(path: str):
    name = os.path.basename(path).lower().replace("-", "_")
    for k in PROMPTS.keys():
        if k.lower() in name:
            return k
    for part in os.path.normpath(path).split(os.sep):
        part = part.lower().replace("-", "_")
        for k in PROMPTS.keys():
            if k.lower() in part:
                return k
    return None

def select_system_prompt(file_path: str, item_task: str):
    k = infer_task_from_path(file_path)
    if k and k in PROMPTS:
        return PROMPTS[k]
    if item_task:
        low = item_task.lower().replace("-", "_")
        for key in PROMPTS.keys():
            if key.lower() in low:
                return PROMPTS[key]
    return None


def make_output_path(output_root: str, input_root: str, file_path: str):
    rel = os.path.relpath(file_path, input_root)
    base, _ = os.path.splitext(rel)
    out_path = os.path.join(output_root, base + ".json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path

async def generate_for_model(
    model_name: str,
    input_dir: str,
    output_dir: str,
    force_rerun: bool = False,
):
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    output_root = os.path.join(output_dir, f"{model_name}_res")
    os.makedirs(output_root, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    for root, _, files in os.walk(input_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            if not fp.endswith(".json"):
                continue
            import json
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    items = json.load(f)
                if not isinstance(items, list):
                    continue
            except Exception:
                continue
            outp = make_output_path(output_root, input_dir, fp)
            
            # Load existing results to identify failed/missing items
            existing_map = {}
            if not force_rerun and os.path.exists(outp):
                try:
                    with open(outp, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        if isinstance(old_data, list):
                            for x in old_data:
                                if "index" in x:
                                    existing_map[x["index"]] = x
                except Exception:
                    print(f"⚠️ Could not read existing output {outp}, starting fresh.")

            items_to_process = []
            for item in items:
                idx = item.get("index")
                if idx is None:
                    # No index, always process
                    items_to_process.append(item)
                    continue
                
                if idx in existing_map:
                    prev_resp = existing_map[idx].get("response", "")
                    # Retry if response is missing, empty, or an error
                    if not prev_resp or prev_resp.startswith("ERROR:"):
                        items_to_process.append(item)
                else:
                    # New item
                    items_to_process.append(item)

            if not items_to_process:
                print(f"✅ All items for {model_name} on {fp} are already completed successfully.")
                continue

            print(f"🔄 Processing {len(items_to_process)}/{len(items)} items for {model_name} on {fp}...")

            tasks = []
            async def run_item(item):
                prompt = item.get("input", "")
                idx = item.get("index", "N/A")
                sys_prompt = select_system_prompt(fp, item.get("task", ""))
                max_retries = 5
                content = ""
                
                for attempt in range(max_retries):
                    try:
                        async with sem:
                            msgs = []
                            if sys_prompt:
                                msgs.append({"role": "system", "content": sys_prompt})
                            msgs.append({"role": "user", "content": prompt})
                            resp = await client.chat.completions.create(
                                model=model_name,
                                messages=msgs,
                                max_tokens=65536,
                                # max_tokens=32768,
                                temperature=0.1
                            )
                        choice = resp.choices[0]
                        content = choice.message.content
                        usage = resp.usage
                        
                        if not content:
                            reason = choice.finish_reason
                            usage_info = f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}" if usage else "usage=N/A"
                            
                            if reason == "refusal":
                                print(f"🚫 Model refused to answer {model_name} on {fp} (index: {idx}). Stopping retries.")
                                content = f"Model refused to answer."
                                break

                            print(f"⚠️ Empty response for {model_name} on {fp} (index: {idx}) (reason: {reason}, {usage_info}). Attempt {attempt + 1}/{max_retries}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            content = f"ERROR: Empty response from model (finish_reason: {reason}, {usage_info})"
                        else:
                            print(f"Response for {model_name} on {fp} (index: {idx}): {content[:50]}...")
                            break # Success
                    except Exception as e:
                        print(f"❌ Error for {model_name} on {fp} (index: {idx}): {e}. Attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            if isinstance(e, RateLimitError) or "429" in str(e):
                                wait_time += 10 # Add extra 60 seconds for rate limits
                                print(f"   ⏳ Rate limit detected. Waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)  # Exponential backoff
                            continue
                        content = f"ERROR: {e}"

                item_out = dict(item)
                item_out["response"] = content
                
                # Log the result
                trunc_prompt = prompt[:10] + "..." if len(prompt) > 10 else prompt
                trunc_response = content[:10] + "..." if len(content) > 10 else content
                logging.info(f"Source: {fp} | Index: {idx} | Model: {model_name} | Prompt: {trunc_prompt} | Response: {trunc_response}")
                
                return item_out
            for it in items_to_process:
                tasks.append(run_item(it))
            new_results = await asyncio.gather(*tasks)
            
            # Merge new results into existing map
            for res in new_results:
                idx = res.get("index")
                if idx is not None:
                    existing_map[idx] = res
            
            # Reconstruct final list in original order
            final_output = []
            for item in items:
                idx = item.get("index")
                if idx is not None and idx in existing_map:
                    final_output.append(existing_map[idx])
                else:
                    # Fallback for items without index or if something went wrong
                    final_output.append(item)

            outp = make_output_path(output_root, input_dir, fp)
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)

async def generate_all(input_dir: str, output_dir: str, force_rerun: bool = False):
    for m in MODELS:
        await generate_for_model(m, input_dir, output_dir, force_rerun=force_rerun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OmicsBench models.")
    parser.add_argument(
        "--input-dir",
        default=os.getenv(
            "OMICS_BENCH_INPUT_DIR",
            os.path.join(os.path.dirname(__file__), "data"),
        ),
        help="Directory containing OmicsBench JSON input files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "OMICS_BENCH_OUTPUT_DIR",
            os.path.join(os.path.dirname(__file__), "results"),
        ),
        help="Directory where per-model generations are written.",
    )
    parser.add_argument("--force", action="store_true", help="Force rerun all tasks, ignoring existing results.")
    args = parser.parse_args()
    
    asyncio.run(generate_all(args.input_dir, args.output_dir, force_rerun=args.force))
