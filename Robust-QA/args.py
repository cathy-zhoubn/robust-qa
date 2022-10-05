import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)

    # For Meta Learning
    parser.add_argument('--meta-datatype', type = str, default ='opt1_singular')
    parser.add_argument('--task-num', type=int, default=5)
    parser.add_argument('--sample-size', type=int, default=126) # TODO change to 127
    parser.add_argument('--n-inner-iter', type=int, default=5)
    parser.add_argument('--inner-lr', type=float, default=1e-1)
    parser.add_argument('--meta-method', type=str, default="reptile_singular")
    
    # For finetune (normally on OOD)
    parser.add_argument('--do-finetune', action='store_true')
    parser.add_argument('--pretrain-model-path', type=str, default="save/baseline-02/checkpoint")
    parser.add_argument('--ft-train-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--ft-val-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--train-ft-dir', type=str, default='datasets/oodomain_train')
    parser.add_argument('--val-ft-dir', type=str, default='datasets/oodomain_val')
    
    args = parser.parse_args()
    return args


def get_aug_dataset_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='datasets/oodomain_train_aug')
    parser.add_argument('--datasets-dir', type=str, default='datasets/oodomain_train')
    parser.add_argument('--datasets-name', type=str, default='relation_extraction') # race, duorc

    parser.add_argument('--run-name', type=str, default='sr') # sr, rd, ri, rs

    parser.add_argument('--alpha', type=float, default= 0.2) 
    parser.add_argument('--naugs', type=int, default=4)

    args = parser.parse_args()
    return args
