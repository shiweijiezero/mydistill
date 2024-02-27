import logging

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import util

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

dataset_name = "EdinburghNLP/xsum"
ds = load_dataset(dataset_name, cache_dir="../data")

logger.info(f"Finish load text from xsum")
logger.info(f"{ds['train']}")

train_dataset = ds["train"]
val_dataset = ds["validation"]
test_dataset = ds["test"]

url = "http://localhost:8000/v1/completions"
headers = {
    "Content-Type": "application/json"
}
prompts = []
outputs = []
labels = []
for item in tqdm(test_dataset,total=len(test_dataset)):
    document = item["document"]
    summary = item["summary"]
    text = document + "\n summary:"
    prompts.append(text)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="gpt2",dtype='bfloat16',max_num_batched_tokens=5)

out = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in out:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    outputs.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

util.UtilClass.eval_xsum(outputs, labels)