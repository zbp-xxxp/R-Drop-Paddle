import logging
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import numpy as np

import paddle
from visualdl import LogWriter
from tqdm import tqdm
import paddle.nn.functional as F
from models.modeling import VisionTransformer
from utils.data_utils import get_loader



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_model(args, model):
    # layer_state_dict = emb.state_dict()
    model_to_save = model.state_dict() #.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pdparams" % args.name)
    paddle.save(model_to_save, model_checkpoint)

def setup(args):
    # Prepare model
    # config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100
    if args.dataset == "imagenet":
        num_classes=1000

    model = VisionTransformer()

    model_PATH = "data/data105204/cifar100-224_checkpoint.pdparams"
    
    model_state_dict = paddle.load(model_PATH)
    model.set_dict(model_state_dict)
    return args, model

class kl_loss(paddle.nn.Layer):
    def __init__(self):
       super(kl_loss, self).__init__()
       self.cross_entropy_loss = paddle.nn.CrossEntropyLoss()

    def forward(self, p, q, label):
        ce_loss = 0.5 * (self.cross_entropy_loss(p, label) + self.cross_entropy_loss(q, label))
        kl_loss = self.compute_kl_loss(p, q)

        # carefully choose hyper-parameters
        loss = ce_loss + 0.3 * kl_loss 

        return loss

    def compute_kl_loss(self, p, q):
        
        p_loss = F.kl_div(F.log_softmax(p, axis=-1), F.softmax(q, axis=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, axis=-1), F.softmax(p, axis=-1), reduction='none')

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2

        return loss

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

def valid(args, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = paddle.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        x, y = batch
        with paddle.no_grad():
            logits = model(x)#[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = paddle.argmax(logits, axis=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    print("accuracy: {}".format(accuracy))

    return accuracy

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.", default="cifar100-test")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100","imagenet"], default="cifar100",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="data/data104692/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=1e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0., type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=100000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--alpha", default=0.6, type=float,
                        help="alpha for kl loss")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    train_loader, test_loader = get_loader(args)
    accuracy = valid(args, model, test_loader)

    # Training


if __name__ == "__main__":
    main()
