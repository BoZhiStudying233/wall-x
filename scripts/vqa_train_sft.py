import os
import shutil
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

from transformers import AutoProcessor
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import save_file as save_safetensors

# Import the installed model class (do not modify model code)
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
    Qwen2_5_VLMoEForAction,
)

@dataclass
class TrainArgs:
    model_path: str
    train_file: str
    image_root: str
    output_dir: str = "./vqa_sft_ckpt"
    batch_size: int = 1
    num_workers: int = 2
    lr: float = 1e-5
    epochs: int = 1
    training_steps: Optional[int] = None  # If set, overrides epochs
    grad_accum_steps: int = 1
    val_file: Optional[str] = None
    validate_every: int = 100
    max_val_samples: Optional[int] = 64
    max_samples: Optional[int] = None
    log_every: int = 10
    precision: str = "bf16"  # choices: ["bf16", "fp16", "fp32"]
    save_format: str = "safetensors"  # choices: ["pt", "safetensors"]
    warmup_ratio: float = 0.03
    warmup_steps: Optional[int] = None
    save_every_steps: int = 5000
    train_log_path: Optional[str] = None
    max_image_side: Optional[int] = None


class VQAJsonlDataset(Dataset):
    """Simple VQA dataset reading JSONL with keys: image, question, answer, question_type."""

    def __init__(self, jsonl_path: str, image_root: str, processor: AutoProcessor, max_image_side: Optional[int] = None):
        self.samples: List[Dict[str, Any]] = []
        self.image_root = image_root
        self.processor = processor
        self.max_image_side = max_image_side

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                image_path = obj.get("image")
                question = obj.get("question")
                answer = obj.get("answer")
                qtype = obj.get("question_type", "vqa")
                if image_path is None or question is None or answer is None:
                    continue
                self.samples.append(
                    {
                        "image": image_path,
                        "question": question,
                        "answer": answer,
                        "dataset_name": qtype,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_path = os.path.join(self.image_root, sample["image"]) if self.image_root else sample["image"]
        image = Image.open(img_path).convert("RGB")
        if self.max_image_side is not None:
            w, h = image.size
            mx = max(w, h)
            if mx > self.max_image_side:
                scale = self.max_image_side / mx
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                image = image.resize((new_w, new_h))

        # Build chat template: user image + text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["question"]},
                ],
            }
        ]

        # Prompt for generation (assistant role header appended)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Target answer text: we keep it minimal (letter) and add eos
        target_text = sample["answer"].strip()
        if len(target_text) == 0:
            target_text = "A"

        eos = self.processor.tokenizer.eos_token or ""
        full_text = prompt_text + target_text + (eos if eos is not None else "")

        # Use the full text + image to get multimodal inputs
        proc = self.processor(text=[full_text], images=[image], return_tensors="pt")

        # Compute label span by tokenizing only the target part (no special tokens)
        answer_ids = self.processor.tokenizer(
            target_text + (eos if eos is not None else ""),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        input_ids = proc["input_ids"][0]
        attention_mask = proc["attention_mask"][0]
        # Build labels: mask everything except the last len(answer_ids) tokens
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[-len(answer_ids) :] = answer_ids

        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # Model expects these for vision
            # Important: keep processor-provided shapes intact (no extra sample dim)
            # - pixel_values shape: (num_image_patches_total, 3, patch, patch)
            # - image_grid_thw shape: (num_images, 3)
            "pixel_values": proc.get("pixel_values") if proc.get("pixel_values") is not None else None,
            "image_grid_thw": proc.get("image_grid_thw") if proc.get("image_grid_thw") is not None else None,
            # Route all tokens to expert 0 by default
            "moe_token_types": torch.zeros_like(input_ids),
            # Use a known multimodal dataset name to avoid KeyError in model loss aggregation
            "dataset_name": "multimodal_VQAv2",
        }

        return item


def vqa_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    # Determine max seq len
    input_ids_list = [b["input_ids"] for b in batch]
    attn_list = [b["attention_mask"] for b in batch]
    labels_list = [b["labels"] for b in batch]
    moe_types_list = [b["moe_token_types"] for b in batch]

    max_len = max(x.size(0) for x in input_ids_list)

    def pad_1d(t: torch.Tensor, pad_value: int):
        if t.size(0) == max_len:
            return t
        return torch.cat(
            [torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype), t], dim=0
        )

    input_ids = torch.stack([pad_1d(x, pad_token_id) for x in input_ids_list], dim=0)
    attention_mask = torch.stack([pad_1d(x, 0) for x in attn_list], dim=0)
    labels = torch.stack([pad_1d(x, -100) for x in labels_list], dim=0)
    moe_token_types = torch.stack([pad_1d(x, 0) for x in moe_types_list], dim=0)

    # Vision fields (stack if present)
    if batch[0]["pixel_values"] is not None:
        # Concatenate along patch dimension across samples
        pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
    else:
        pixel_values = None
    if batch[0]["image_grid_thw"] is not None:
        # Concatenate images across samples: shape (sum(num_images), 3)
        image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    else:
        image_grid_thw = None

    dataset_names = [b.get("dataset_name", "multimodal_VQAv2") for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "moe_token_types": moe_token_types,
        "dataset_names": dataset_names,
    }


def get_dtype(precision: str):
    prec = precision.lower()
    if prec == "bf16":
        return torch.bfloat16
    if prec == "fp16":
        return torch.float16
    return torch.float32


def train(args: TrainArgs):
    # DDP or Single-GPU setup
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if ddp else 0

    if ddp and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main = (not ddp) or (dist.get_rank() == 0)

    # Load processor and model
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'  # Use left padding for generation tasks
    try:
        model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    except TypeError:
        model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)

    # # 冻结 visual backbone
    # for name, param in model.named_parameters():
    #     if 'visual' in name:  # 冻结包含 'visual' 的所有参数
    #         param.requires_grad = False
    #         if is_main:
    #             print(f"Frozen parameter: {name}")

    dtype = get_dtype(args.precision)
    if device.startswith("cuda"):
        model = model.to(device, dtype=dtype)
    else:
        model = model.to(device)

    # Wrap with DDP if needed
    if ddp and torch.cuda.is_available():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Dataset + loader
    dataset = VQAJsonlDataset(args.train_file, args.image_root, processor, max_image_side=args.max_image_side)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: args.max_samples]

    # Sampler for DDP
    if ddp:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(dataset, shuffle=True)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    collate = lambda batch: vqa_collate_fn(batch, pad_token_id=processor.tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )

    # Optimizer (lr is the max lr; warmup + cosine will scale it)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    model.train()
    global_step = 0
    scaler = None  # Optional: leave off grad scaler unless using fp16
    
    # Determine total training steps: use training_steps if specified, else epochs
    steps_per_epoch = max(1, math.ceil(len(loader) / max(1, args.grad_accum_steps)))
    if args.training_steps is not None:
        total_update_steps = args.training_steps
        num_epochs = math.ceil(args.training_steps / steps_per_epoch)
        if is_main:
            print(f"Training for {total_update_steps} steps (~{num_epochs} epochs)")
    else:
        total_update_steps = steps_per_epoch * args.epochs
        num_epochs = args.epochs
        if is_main:
            print(f"Training for {num_epochs} epochs ({total_update_steps} steps)")
    
    warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else int(total_update_steps * args.warmup_ratio)
    )

    # Use transformers' cosine schedule with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps
    )

    # Setup training log file
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = args.train_log_path or os.path.join(args.output_dir, "training_log.txt")
        log_fh = open(log_path, "a", encoding="utf-8")
    else:
        log_fh = None

    def run_validation() -> Optional[float]:
        if not args.val_file:
            return None
        eval_model = model.module if isinstance(model, DDP) else model
        eval_model.eval()
        total = 0
        correct = 0
        # Lightweight on-the-fly loop, sample-by-sample to keep shapes simple
        with open(args.val_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.max_val_samples is not None and i >= args.max_val_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                img_rel = obj.get("image")
                question = obj.get("question")
                answer = str(obj.get("answer", "")).strip()
                if not img_rel or not question or not answer:
                    continue
                # Build message and process
                img_path = os.path.join(args.image_root, img_rel) if args.image_root else img_rel
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                proc = processor(text=[prompt], images=[image], return_tensors="pt")
                inputs = {k: v for k, v in proc.items()}
                inputs["input_ids"] = inputs["input_ids"].to(device)
                inputs["attention_mask"] = inputs["attention_mask"].to(device)
                if inputs.get("pixel_values") is not None:
                    inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=dtype)
                if inputs.get("image_grid_thw") is not None:
                    inputs["image_grid_thw"] = inputs["image_grid_thw"].to(device)
                # Model-specific required fields
                inputs["moe_token_types"] = torch.zeros_like(inputs["input_ids"]).to(device)
                inputs["dataset_names"] = ["multimodal_VQAv2"]

                gen_params = {
                    "max_new_tokens": 16,
                    "do_sample": False,
                    "eos_token_id": processor.tokenizer.eos_token_id,
                    "pad_token_id": processor.tokenizer.pad_token_id,
                }
                with torch.no_grad():
                    out_ids = eval_model.generate(**inputs, **gen_params)
                # Slice generated part
                gen_only = out_ids[0, inputs["input_ids"].shape[1] :]
                text = processor.batch_decode([gen_only], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                # Extract first choice letter A/B/C/D present
                pred = None
                for ch in ["A", "B", "C", "D"]:
                    if ch in text:
                        pred = ch
                        break
                if pred is None and len(text.strip()) > 0:
                    # Fallback: first uppercase alpha
                    for c in text:
                        if c.isalpha():
                            pred = c.upper()
                            break
                if pred is not None and pred == answer.strip().upper():
                    correct += 1
                if answer in ["A", "B", "C", "D"]:
                    total += 1
        acc = (correct / total) if total > 0 else 0.0
        if is_main:
            print(f"[Validation] samples={total} acc={acc:.4f}")
        eval_model.train()
        return acc

    for epoch in range(num_epochs):
        if ddp and 'train_sampler' in locals() and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            # Check if we've reached the target training steps
            if args.training_steps is not None and global_step >= args.training_steps:
                if is_main:
                    print(f"Reached target training_steps={args.training_steps}, stopping training.")
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            moe_token_types = batch["moe_token_types"].to(device)
            dataset_names = batch["dataset_names"]

            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=dtype)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            # Mixed precision autocast to reduce memory
            use_amp = device.startswith("cuda") and dtype in (torch.bfloat16, torch.float16)
            amp_ctx = torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext()

            # Avoid gradient sync on non-final micro-steps under DDP
            sync_ctx = nullcontext()
            if ddp and isinstance(model, DDP) and ((step + 1) % max(1, args.grad_accum_steps) != 0):
                sync_ctx = model.no_sync()

            with sync_ctx:
                with amp_ctx:
                    outputs = model(
                        mode="train",
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        moe_token_types=moe_token_types,
                        dataset_names=dataset_names,
                        return_dict=True,
                    )

            loss = outputs.loss
            loss = loss / max(1, args.grad_accum_steps)
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main and global_step % args.log_every == 0:
                    ce = (
                        outputs.cross_entropy_loss.detach().item()
                        if outputs.cross_entropy_loss is not None
                        else float("nan")
                    )
                    cur_lr = optimizer.param_groups[0]["lr"]
                    msg = f"epoch {epoch} step {global_step} | loss={loss.item():.4f} ce={ce:.4f} lr={cur_lr:.6e}"
                    print(msg)
                    if log_fh is not None:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        log_fh.write(f"[{ts}] {msg}\n")
                        log_fh.flush()

                # Periodic validation on global steps
                if args.val_file and (global_step % max(1, args.validate_every) == 0):
                    if ddp:
                        dist.barrier()
                    acc = run_validation()
                    if log_fh is not None:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        log_fh.write(f"[{ts}] validation step={global_step} acc={acc}\n")
                        log_fh.flush()
                    if ddp:
                        dist.barrier()

                # Periodic checkpoint save by steps
                if is_main and (global_step % max(1, args.save_every_steps) == 0):
                    to_save = model.module if isinstance(model, DDP) else model
                    if args.save_format == "safetensors":
                        save_path = os.path.join(args.output_dir, f"model_step_{global_step}.safetensors")
                        state_dict = {k: v.detach().to("cpu") for k, v in to_save.state_dict().items()}
                        save_safetensors(state_dict, save_path)
                    else:
                        save_path = os.path.join(args.output_dir, f"pytorch_model_step_{global_step}.pt")
                        torch.save(to_save.state_dict(), save_path)
                    print(f"Saved checkpoint to {save_path}")
                    # one-time artifacts
                    src_cfg = os.path.join(args.model_path, "config.json")
                    dst_cfg = os.path.join(args.output_dir, "config.json")
                    if os.path.exists(src_cfg) and not os.path.exists(dst_cfg):
                        try:
                            shutil.copy(src_cfg, dst_cfg)
                            print(f"Copied config.json to {dst_cfg}")
                        except Exception as e:
                            print(f"Warn: failed to copy config.json: {e}")
                    proc_marker = os.path.join(args.output_dir, "preprocessor_config.json")
                    try:
                        if not os.path.exists(proc_marker):
                            processor.save_pretrained(args.output_dir)
                            print(f"Saved processor to {args.output_dir}")
                    except Exception as e:
                        print(f"Warn: failed to save processor: {e}")
        
        # Break outer loop if we've reached target training steps
        if args.training_steps is not None and global_step >= args.training_steps:
            break

    if ddp:
        dist.barrier()

    # Close log file
    if log_fh is not None:
        log_fh.close()


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Prototype VQA SFT training loop (custom)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./vqa_sft_ckpt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (ignored if --training_steps is set)")
    parser.add_argument("--training_steps", type=int, default=None, help="Total training steps (overrides --epochs if set)")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--max_val_samples", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save_format", type=str, default="safetensors", choices=["pt", "safetensors"])
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--save_every_steps", type=int, default=5000)
    parser.add_argument("--train_log_path", type=str, default=None)
    parser.add_argument("--max_image_side", type=int, default=None)

    ns = parser.parse_args()
    return TrainArgs(
        model_path=ns.model_path,
        train_file=ns.train_file,
        image_root=ns.image_root,
        output_dir=ns.output_dir,
        batch_size=ns.batch_size,
        num_workers=ns.num_workers,
        lr=ns.lr,
        epochs=ns.epochs,
        training_steps=ns.training_steps,
        grad_accum_steps=ns.grad_accum_steps,
        val_file=ns.val_file,
        validate_every=ns.validate_every,
        max_val_samples=ns.max_val_samples,
        max_samples=ns.max_samples,
        log_every=ns.log_every,
        precision=ns.precision,
        save_format=ns.save_format,
        warmup_ratio=ns.warmup_ratio,
        warmup_steps=ns.warmup_steps,
        save_every_steps=ns.save_every_steps,
        train_log_path=ns.train_log_path,
        max_image_side=ns.max_image_side,
    )


if __name__ == "__main__":
    train(parse_args())
