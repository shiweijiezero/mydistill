import os

from datasets import load_dataset
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

ds = load_dataset(path="hails/mmlu_no_train",name="all")

# names = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics',
#          'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology',
#          'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology',
#          'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition',
#          'global_facts', 'machine_learning', 'security_studies', 'public_relations',
#          'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine',
#          'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies',
#          'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics',
#          'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law',
#          'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous',
#          'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics',
#          'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics',
#          'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy',
#          'college_biology']
# for name in tqdm(names, total=len(names)):
#     ds = load_dataset(path="hails/mmlu_no_train", name=name, )
