# Automatically generated using ../scripts/build_datasets_info.py. 

from datasets import ClassLabel, Value

valid_task_names=['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg', 'cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'wnli', 'ax', 'imdb', 'hans', 'squad', 'squad_v2']

datasets_info = {

	"boolq": {
		"load_dataset_args": ('super_glue', 'boolq'),
		"sentence_keys": ['question', 'passage'],
		"label": ClassLabel(num_classes=2, names=('False', 'True'), id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [3245, 9427, 3270],
	},

	"cb": {
		"load_dataset_args": ('super_glue', 'cb'),
		"sentence_keys": ['premise', 'hypothesis'],
		"label": ClassLabel(num_classes=3, names=['entailment', 'contradiction', 'neutral'], id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [250, 250, 56],
	},

	"copa": {
		"load_dataset_args": ('super_glue', 'copa'),
		"sentence_keys": ['premise', 'choice1', 'choice2', 'question'],
		"label": ClassLabel(num_classes=2, names=['choice1', 'choice2'], id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [500, 400, 100],
	},

	"multirc": {
		"load_dataset_args": ('super_glue', 'multirc'),
		"sentence_keys": ['paragraph', 'question', 'answer'],
		"label": ClassLabel(num_classes=2, names=('False', 'True'), id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [9693, 27243, 4848],
	},

	"record": {
		"load_dataset_args": ('super_glue', 'record'),
		"sentence_keys": ['passage', 'query', 'entities', 'entity_spans', 'answers'],
		"label": None,
		"splits": ['train', 'validation', 'test'],
		"splits_num_examples": [100730, 10000, 10000],
	},

	"rte": {
		"load_dataset_args": ('glue', 'rte'),
		"sentence_keys": ['sentence1', 'sentence2'],
		"label": ClassLabel(num_classes=2, names=['entailment', 'not_entailment'], id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [3000, 2490, 277],
	},

	"wic": {
		"load_dataset_args": ('super_glue', 'wic'),
		"sentence_keys": ['word', 'sentence1', 'sentence2', 'start1', 'start2', 'end1', 'end2'],
		"label": ClassLabel(num_classes=2, names=('False', 'True'), id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [1400, 5428, 638],
	},

	"wsc": {
		"load_dataset_args": ('super_glue', 'wsc'),
		"sentence_keys": ['text', 'span1_index', 'span2_index', 'span1_text', 'span2_text'],
		"label": ClassLabel(num_classes=2, names=('False', 'True'), id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [146, 554, 104],
	},

	"wsc.fixed": {
		"load_dataset_args": ('super_glue', 'wsc.fixed'),
		"sentence_keys": ['text', 'span1_index', 'span2_index', 'span1_text', 'span2_text'],
		"label": ClassLabel(num_classes=2, names=('False', 'True'), id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [146, 554, 104],
	},

	"axb": {
		"load_dataset_args": ('super_glue', 'axb'),
		"sentence_keys": ['sentence1', 'sentence2'],
		"label": ClassLabel(num_classes=2, names=['entailment', 'not_entailment'], id=None),
		"splits": ['test'],
		"splits_num_examples": [1104],
	},

	"axg": {
		"load_dataset_args": ('super_glue', 'axg'),
		"sentence_keys": ['premise', 'hypothesis'],
		"label": ClassLabel(num_classes=2, names=['entailment', 'not_entailment'], id=None),
		"splits": ['test'],
		"splits_num_examples": [356],
	},

	"cola": {
		"load_dataset_args": ('glue', 'cola'),
		"sentence_keys": ['sentence'],
		"label": ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], id=None),
		"splits": ['train', 'validation', 'test'],
		"splits_num_examples": [8551, 1043, 1063],
	},

	"sst2": {
		"load_dataset_args": ('glue', 'sst2'),
		"sentence_keys": ['sentence'],
		"label": ClassLabel(num_classes=2, names=['negative', 'positive'], id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [1821, 67349, 872],
	},

	"mrpc": {
		"load_dataset_args": ('glue', 'mrpc'),
		"sentence_keys": ['sentence1', 'sentence2'],
		"label": ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [1725, 3668, 408],
	},

	"qqp": {
		"load_dataset_args": ('glue', 'qqp'),
		"sentence_keys": ['question1', 'question2'],
		"label": ClassLabel(num_classes=2, names=['not_duplicate', 'duplicate'], id=None),
		"splits": ['train', 'validation', 'test'],
		"splits_num_examples": [363846, 40430, 390965],
	},

	"stsb": {
		"load_dataset_args": ('glue', 'stsb'),
		"sentence_keys": ['sentence1', 'sentence2'],
		"label": Value(dtype='float32', id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [1379, 5749, 1500],
	},

	"mnli": {
		"load_dataset_args": ('glue', 'mnli'),
		"sentence_keys": ['premise', 'hypothesis'],
		"label": ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'], id=None),
		"splits": ['test_matched', 'test_mismatched', 'train', 'validation_matched', 'validation_mismatched'],
		"splits_num_examples": [9796, 9847, 392702, 9815, 9832],
	},

	"mnli_mismatched": {
		"load_dataset_args": ('glue', 'mnli_mismatched'),
		"sentence_keys": ['premise', 'hypothesis'],
		"label": ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'], id=None),
		"splits": ['test', 'validation'],
		"splits_num_examples": [9847, 9832],
	},

	"mnli_matched": {
		"load_dataset_args": ('glue', 'mnli_matched'),
		"sentence_keys": ['premise', 'hypothesis'],
		"label": ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'], id=None),
		"splits": ['test', 'validation'],
		"splits_num_examples": [9796, 9815],
	},

	"qnli": {
		"load_dataset_args": ('glue', 'qnli'),
		"sentence_keys": ['question', 'sentence'],
		"label": ClassLabel(num_classes=2, names=['entailment', 'not_entailment'], id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [5463, 104743, 5463],
	},

	"wnli": {
		"load_dataset_args": ('glue', 'wnli'),
		"sentence_keys": ['sentence1', 'sentence2'],
		"label": ClassLabel(num_classes=2, names=['not_entailment', 'entailment'], id=None),
		"splits": ['test', 'train', 'validation'],
		"splits_num_examples": [146, 635, 71],
	},

	"ax": {
		"load_dataset_args": ('glue', 'ax'),
		"sentence_keys": ['premise', 'hypothesis'],
		"label": ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'], id=None),
		"splits": ['test'],
		"splits_num_examples": [1104],
	},

	"imdb": {
		"load_dataset_args": ('imdb', 'plain_text'),
		"sentence_keys": ['text'],
		"label": ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
		"splits": ['train', 'test', 'unsupervised'],
		"splits_num_examples": [25000, 25000, 50000],
	},

	"hans": {
		"load_dataset_args": ('hans', 'plain_text'),
		"sentence_keys": ['premise', 'hypothesis', 'parse_premise', 'parse_hypothesis', 'binary_parse_premise', 'binary_parse_hypothesis', 'heuristic', 'subcase', 'template'],
		"label": ClassLabel(num_classes=2, names=['entailment', 'non-entailment'], id=None),
		"splits": ['train', 'validation'],
		"splits_num_examples": [30000, 30000],
	},

	"squad": {
		"load_dataset_args": ('squad', 'plain_text'),
		"sentence_keys": ['title', 'context', 'question', 'answers'],
		"label": None,
		"splits": ['train', 'validation'],
		"splits_num_examples": [87599, 10570],
	},

	"squad_v2": {
		"load_dataset_args": ('squad_v2', 'squad_v2'),
		"sentence_keys": ['title', 'context', 'question', 'answers'],
		"label": None,
		"splits": ['train', 'validation'],
		"splits_num_examples": [130319, 11873],
	},
}
