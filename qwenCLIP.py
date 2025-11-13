import os
import socket
import subprocess
import numpy as np
import torch
import shutil
import torchvision
import argparse
import datetime
import time
import pickle
import torch.nn.functional as F
from torch import nn



from torch import Tensor
from transformers import AutoTokenizer, AutoModel

torch.autograd.set_detect_anomaly(True)
import transformers


import sys
sys.path.append('/gpfsdswork/projects/rech/dvj/uyk23wk/ConVIRT/clip/lib/python3.6/site-packages/clip')
from clip import *

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# ConVIRT
import clip
from convirt.modules.transformations import TransformsConVIRT
from convirt.modules.sync_batchnorm import convert_model
from convirt.modules.dataloader import CLRDataset,MTDataset
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from model import load_optimizer
from utils import yaml_config_hook

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def save_model(args, model, optimizer, best=False):
    if best:
        out = os.path.join(args.model_path, "best_checkpoint_{}.pth".format(args.current_epoch))
    else:
        out = os.path.join(args.model_path, "checkpoint_{}.pth".format(args.current_epoch))
    torch.save(model, out)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad != None:
            p.grad.data = p.grad.data.float()
def convert_models_to_mix(model):
    clip.model.convert_weights(model)

def make_prompt(text, modality=None):
    return f"Modality: {modality}\nQuery: {text}"




def train(args, train_loader, model, tokenizer, optimizer,writer):
    loss_epoch = 0
    for step, (images, texts, x_n) in enumerate(train_loader):
        optimizer.zero_grad()
        # modality=[img_dict[name] for name in x_n]
        # prompt=[make_prompt(t,modality[i]) for i,t in enumerate(texts)]
        x_v = images.to(args.device)
        v=model.visual(x_v)
        labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
        loss_CE = torch.nn.CrossEntropyLoss()
        txt=['\nQuery:'+t for t in texts]

        # --- MODIFIED BLOCK START ---
        # Tokenize main text
        batch_dict = tokenizer(
            txt,
            padding=True,
            truncation=True,
            max_length=1024, 
            return_tensors="pt",
        )
        batch_dict.to(args.device)
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']


        with torch.no_grad():
            # Get word embeddings for text (model.transformer.get_input_embeddings() is frozen)
            inputs_embeds = model.transformer.get_input_embeddings()(input_ids).float()
            batch_size = inputs_embeds.shape[0]

            # 1. Get STATIC prompt embeddings (frozen)
            static_prompt_embeds = model.transformer.get_input_embeddings()(model.static_prompt_tokens).expand(batch_size, -1, -1).float()
            
            # 2. Get LEARNABLE prompt embeddings (trainable)
            learnable_prompt_embeds = model.learnable_prompt_embeddings.to(inputs_embeds.device).expand(batch_size, -1, -1)
            
            # 3. Concatenate all embeddings: [STATIC, LEARNABLE, TEXT]
            combined_embeds = torch.cat((static_prompt_embeds, learnable_prompt_embeds, inputs_embeds), dim=1)

            # Adjust attention mask
            static_attention_mask = torch.ones(batch_size, model.num_static_tokens, device=args.device, dtype=torch.long)
            learnable_attention_mask = torch.ones(batch_size, model.num_learnable_tokens, device=args.device, dtype=torch.long)
            combined_attention_mask = torch.cat((static_attention_mask, learnable_attention_mask, attention_mask), dim=1)

            # Forward pass with combined embeddings
            outputs = model.transformer(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
            
            # Pool the last token of the *entire sequence*
            embeddings = last_token_pool(outputs.last_hidden_state, combined_attention_mask)
            # --- MODIFIED BLOCK END ---

        if args.llm_proj:
            u=model.llm_proj(embeddings.float())
        else:
            u = embeddings[..., :512] 

        # normalized features
        image_features = v / v.norm(dim=1, keepdim=True)
        text_features = u / u.norm(dim=1, keepdim=True).float()

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        print(f"Step {step}: GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    
        loss_v = loss_CE(logits_per_image,labels)
        loss_t = loss_CE(logits_per_text,labels)
        loss=loss_v+loss_t

        loss.backward()
        # convert_models_to_fp32(model)
        optimizer.step()
        # convert_models_to_mix(model)

        if args.nr == 0 and step % 1000 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
        del batch_dict, input_ids, attention_mask, outputs, embeddings
    return loss_epoch


def validate(args, val_loader, model, tokenizer, optimizer,writer):
    with torch.no_grad():
        model.eval()
        loss_epoch = 0
        for step, (images, texts, x_n) in enumerate(val_loader):
            x_v = images.to(args.device)
            # modality=[img_dict[name] for name in x_n]
            # prompt=[make_prompt(t,modality[i]) for i,t in enumerate(x_u)]
            v=model.visual(x_v)
            labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
            loss_CE = torch.nn.CrossEntropyLoss()
            txt=['\nQuery:'+t for t in texts]

             # --- MODIFIED BLOCK START ---
            # Tokenize main text
            batch_dict = tokenizer(
                txt,
                padding=True,
                truncation=True,
                max_length=1024, 
                return_tensors="pt",
            )
            batch_dict.to(args.device)
            input_ids = batch_dict['input_ids']
            attention_mask = batch_dict['attention_mask']

            # Get word embeddings for text (frozen)
            inputs_embeds = model.transformer.get_input_embeddings()(input_ids).float()
            batch_size = inputs_embeds.shape[0]

            # 1. Get STATIC prompt embeddings (frozen)
            static_prompt_embeds = model.transformer.get_input_embeddings()(model.static_prompt_tokens).expand(batch_size, -1, -1).float()
            
            # 2. Get LEARNABLE prompt embeddings (trainable)
            learnable_prompt_embeds = model.learnable_prompt_embeddings.to(inputs_embeds.device).expand(batch_size, -1, -1)
            
            # 3. Concatenate all embeddings: [STATIC, LEARNABLE, TEXT]
            combined_embeds = torch.cat((static_prompt_embeds, learnable_prompt_embeds, inputs_embeds), dim=1)

            # Adjust attention mask
            static_attention_mask = torch.ones(batch_size, model.num_static_tokens, device=args.device, dtype=torch.long)
            learnable_attention_mask = torch.ones(batch_size, model.num_learnable_tokens, device=args.device, dtype=torch.long)
            combined_attention_mask = torch.cat((static_attention_mask, learnable_attention_mask, attention_mask), dim=1)

            # Forward pass with combined embeddings
            outputs = model.transformer(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
            
            # Pool
            embeddings = last_token_pool(outputs.last_hidden_state, combined_attention_mask)
            # --- MODIFIED BLOCK END ---

            if args.llm_proj:
                u=model.llm_proj(embeddings.float())
            else:
                u = embeddings[..., :512] 


            # normalized features
            image_features = v / v.norm(dim=1, keepdim=True)
            text_features = u / u.norm(dim=1, keepdim=True).float()

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()



            loss_v = loss_CE(logits_per_image,labels)
            loss_t = loss_CE(logits_per_text,labels)
            loss=loss_v+loss_t


            loss_epoch += loss.item()


    model.train()
    return loss_epoch


def main(gpu, args):
    # number of nodes / node ID
    job_name = os.environ['SLURM_JOB_NAME']
    job_id = os.environ['SLURM_JOB_ID']
    log_filename = job_name + job_id + '.loss'


    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])

    # local rank on the current node / global rank
    local_rank = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    # number of processes / GPUs per node
    world_size = int(os.environ['SLURM_NTASKS'])
    n_gpu_per_node = world_size // n_nodes

    # define master address and master port
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    master_addr = hostnames.split()[0].decode('utf-8')

    # set environment variables for 'env://'
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(29500)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)

    # define whether this is the master process / if we are in distributed mode
    is_master = node_id == 0 and local_rank == 0
    multi_node = n_nodes > 1
    multi_gpu = world_size > 1

    # summary
    PREFIX = "%i - " % global_rank
    print(PREFIX + "Number of nodes: %i" % n_nodes)
    print(PREFIX + "Node ID        : %i" % node_id)
    print(PREFIX + "Local rank     : %i" % local_rank)
    print(PREFIX + "Global rank    : %i" % global_rank)
    print(PREFIX + "World size     : %i" % world_size)
    print(PREFIX + "GPUs per node  : %i" % n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(is_master))
    print(PREFIX + "Multi-node     : %s" % str(multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(multi_gpu))
    print(PREFIX + "Hostname       : %s\n" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('torch.cuda.is_available:',torch.cuda.is_available())


    # initialize model
    if is_master:
        print("Initializing model... ", end="", flush=True)
    model, preprocess = clip.load(args.resnet.split("@")[-1], device=args.device, jit=False)
    model = model.float()
    llm_dir='/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/'
    tokenizer=clip.tokenize
    if not args.clip:
        tokenizer = AutoTokenizer.from_pretrained(llm_dir+args.qwen, padding_side='left')
        model.transformer = AutoModel.from_pretrained(llm_dir+args.qwen,load_in_8bit=True)

        # --- MODIFIED/ADDED START ---
        static_prompt_text = "Instruct: "
        learnable_prompt_text = "Create a dense embedding that captures the medical findings for image retrieval"
        
        if is_master:
            print(f"\nInitializing static prompt: '{static_prompt_text}'")
            print(f"Initializing learnable prompt with: '{learnable_prompt_text}'")
        
        # Tokenize static part (without special tokens) and register as buffer
        static_prompt_tokens = tokenizer(
            static_prompt_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids.to(args.device)
        model.register_buffer('static_prompt_tokens', static_prompt_tokens)
        
        # Tokenize learnable part (without special tokens)
        learnable_prompt_tokens = tokenizer(
            learnable_prompt_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids.to(args.device)
        
        # Get initial embeddings for the learnable part
        with torch.no_grad():
            initial_learnable_embeddings = model.transformer.get_input_embeddings()(learnable_prompt_tokens).float()

        # Create the learnable parameter
        model.learnable_prompt_embeddings = nn.Parameter(initial_learnable_embeddings)
        

        # Store prompt lengths
        model.num_static_tokens = static_prompt_tokens.shape[1]
        model.num_learnable_tokens = learnable_prompt_tokens.shape[1]
        model.total_prompt_tokens = model.num_static_tokens + model.num_learnable_tokens        
        if is_master:
            print(f"Static prompt tokens: {model.num_static_tokens}, Learnable prompt tokens: {model.num_learnable_tokens}, Total: {model.total_prompt_tokens}")

        
        # Get hidden dim from model config
        qwen_dim = model.transformer.config.hidden_size
        if is_master:
            print(f"Detected Qwen hidden dimension: {qwen_dim}")


        hidden = 2048 if qwen_dim==4096 else 1024
        if args.llm_proj==2:
            model.llm_proj = nn.Sequential(
            nn.Linear(qwen_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 512)
        )
        elif args.llm_proj==1:
            model.llm_proj=nn.Linear(qwen_dim,512)
        
        for p in model.transformer.parameters():
            p.requires_grad = False


    model.to(args.device)

    train_fonction = train
    validate_fonction = validate


    if is_master:
        print("Image encoder:", args.resnet,
            "\tPretrained:", args.pretrain)
        print("Text encoder:", args.bert.split("/")[-1],
            "\tFreezed layers:", args.freeze_layers, "\n")
        print("Tokenizer:",tokenizer)


    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model, lr=float(args.lr))


    if "clip" in args.resnet or "RN50" in args.resnet:
        transform = preprocess

    train_dataset = CLRDataset(csv_file=args.csv_file,
                               root_dir=args.root_dir,
                               transform=transform,
                               clip = ("clip" in args.resnet or "RN50" in args.resnet)
                               )
    print('training set:',args.csv_file,'len(dataset):',train_dataset.__len__())

    val_dataset = CLRDataset(csv_file=args.val_csv_file,
                             root_dir=args.val_root_dir,
                             transform=transform,
                             clip = ("clip" in args.resnet or "RN50" in args.resnet)
                             )
    print('validation set:',args.val_csv_file,'len(csv):',val_dataset.__len__())

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers
    )

    if is_master:
        print("[DONE]\n")

    # print(list(train_loader)[0])

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    if is_master:
        print("STARTING TRAINING")
        print('Start Time =', datetime.datetime.now().strftime("%H:%M:%S"), '\n')

    t0 = time.time()
    args.global_step = 0
    args.current_epoch = 0
    best_val_loss = np.inf


    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train_fonction(args, train_loader, model, tokenizer, optimizer,writer)

        if args.nr == 0 and scheduler:
            scheduler.step()


        if args.nr == 0 and is_master:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            val_loss = validate_fonction(args, val_loader, model, tokenizer, optimizer,writer)
            if val_loss < best_val_loss:
                out = os.path.join(args.model_path, "ViT_best_epoch_{}_{}.tar".format(args.current_epoch, val_loss / len(val_loader)))
                torch.save(model.visual, out)
                # save_model(args, model, optimizer, best=True)
                best_val_loss = val_loss
            else:
                save_model(args, model.visual, optimizer, best=False)

            epoch_counter = epoch - args.start_epoch
            elapsed = time.time() - t0
            epoch_time = elapsed/(epoch_counter+1)
            remaining = (args.epochs - (epoch_counter+1))*epoch_time
            remaining = str(datetime.timedelta(seconds=round(remaining)))
            elapsed = str(datetime.timedelta(seconds=round(elapsed)))
            print(f'Epoch {epoch_counter+1}/{args.epochs} [{elapsed}<{remaining}, {round(epoch_time, 2)}s/epoch] {round((epoch_counter+1)/args.epochs*100, 1)}% loss: {loss_epoch / len(train_loader)}\t val_loss: {val_loss / len(val_loader)} lr: {lr}')

            with open(os.path.join(args.log_loss_dir,log_filename), 'a') as f:
                f.write(str(loss_epoch / len(train_loader)) + ',' + str(val_loss / len(val_loader)) + '\n')

            args.current_epoch += 1
            # if args.track:
            #     out = os.path.join(args.model_path, "ViT_checkpoint_{}_{}.tar".format(args.current_epoch, val_loss / len(val_loader)))
            #     torch.save(model.model.visual, out)

    # end training
    if is_master:
        save_model(args, model.model.visual, optimizer)
    writer.close()


if __name__ == "__main__":
    t1=time.time()
    parser = argparse.ArgumentParser(description="ConVIRT")
    config = yaml_config_hook("./config/config_qwenCLIP_modality_ViT16.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    shutil.copy('./config/config_qwenCLIP_modality_ViT16.yaml',os.path.join(args.model_path, 'config.yaml'))

    with open('/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/rocov2img_modality.pkl', 'rb') as f:
        img_dict = pickle.load(f)

    print("args.model_path",args.model_path)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    print("Device:", args.device)

    main(0, args)