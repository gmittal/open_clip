from datasets import load_dataset
import sys
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mnli', help='HF dataset name')
    parser.add_argument('--prompt1-key', type=str, default='premise')
    parser.add_argument('--prompt2-key', type=str, default='hypothesis')
    parser.add_argument('--train-key', type=str, default='train')
    parser.add_argument('--validation-key', type=str, default='validation_matched')
    parser.add_argument('--out-path', type=str)
    parser.add_argument('--separator-token', type=str, default='<end_of_text>')

    args = parser.parse_args(args)
    return args


def main(argv):
    args = parse_args(argv)
    dataset = load_dataset('glue', args.task)
    combine_func = None
    if args.task == 'mnli':
        def combine_func(example):
            example['sentence'] = example['premise'] + args.separator_token + example['hypothesis']
            return example
    elif args.task == 'mrpc':
        def combine_func(example):
            example['sentence'] = example['sentence1'] + args.separator_token + example['sentence2']
            return example
    
    combined_dataset = dataset.map(combine_func)
    combined_dataset.save_to_disk(args.out_path)
    

if __name__ == '__main__':
    main(sys.argv[1:])
