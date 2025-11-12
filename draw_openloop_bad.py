# init_model.py
import os
import torch

from tqdm import tqdm

import os
import yaml
import torch
import matplotlib.pyplot as plt
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs
from transformers import AutoProcessor
from wall_x.trainer.qwen_vl_act_trainer import QwenVlAct_Trainer
from wall_x.model.qwen2_5_based import Qwen2_5_VLMoEForAction, Qwen2_5_VLConfig


# === 按你的工程实际路径替换这些导入 ===
# from your_package.models import Qwen2_5_VLConfig, Qwen2_5_VLMoEForAction
# 如果在同目录或模块名不同，请改成正确的 import
from wall_x.model.qwen2_5_based import Qwen2_5_VLMoEForAction, Qwen2_5_VLConfig

# 可选：如果你用 Accelerate 管理设备
try:
    from accelerate import Accelerator
except Exception:
    Accelerator = None


@torch.no_grad()
def load_qwen_pretrain_weight(model, pretrain_weight_path: str):
    """
    与训练脚本逻辑一致，但在加载前先把模型的vocab size临时调整到ckpt的 embed/lm_head 尺寸，
    以避免 size mismatch；加载后再按当前processor的tokenizer长度做最终 resize。
    """
    import os
    from safetensors.torch import load_file

    # 1) 读并合并 safetensors
    weight_files = sorted(
        [f for f in os.listdir(pretrain_weight_path) if f.endswith(".safetensors")]
    )
    merged_weights = {}
    for weight_file in weight_files:
        file_path = os.path.join(pretrain_weight_path, weight_file)
        weights = load_file(file_path)
        merged_weights.update(weights)

    # 2) MoE 重命名
    renamed_weights = {}
    for key, value in merged_weights.items():
        if key.startswith("model.layers") and "mlp." in key and getattr(model.config, "mlp_moe", False):
            layer_num = key.split(".layers.")[1].split(".mlp")[0]
            new_key = key.replace(f"layers.{layer_num}.mlp.", f"layers.{layer_num}.moe.experts.0.")
            renamed_weights[new_key] = value
        elif key.startswith("model.layers") and "self_attn." in key and getattr(model.config, "attention_moe", False):
            layer_num = key.split(".layers.")[1].split(".self_attn")[0]
            proj_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
            hit = False
            for proj in proj_types:
                if f".{proj}." in key or key.endswith(f".{proj}.weight") or key.endswith(f".{proj}.bias"):
                    new_key = key.replace(
                        f"layers.{layer_num}.self_attn.{proj}",
                        f"layers.{layer_num}.self_attn.{proj}_experts.0",
                    )
                    renamed_weights[new_key] = value
                    hit = True
                    break
            if not hit:
                renamed_weights[key] = value
        else:
            renamed_weights[key] = value

    # 3) 处理 vocab 大小不一致：先把当前模型 resize 到 ckpt 的 vocab 再加载
    # 可能的 key 名（不同实现会有差异）
    embed_keys = [
        "model.embed_tokens.weight",          # 常见
        "transformer.embed_tokens.weight",    # 备选
        "embed_tokens.weight",                # 备选
    ]
    lm_head_keys = [
        "lm_head.weight",
        "model.lm_head.weight",
    ]

    ckpt_embed_key = next((k for k in embed_keys if k in renamed_weights), None)
    ckpt_lm_head_key = next((k for k in lm_head_keys if k in renamed_weights), None)

    if ckpt_embed_key is None and ckpt_lm_head_key is None:
        # 没有发现这俩权重，直接正常加载
        err = model.load_state_dict(renamed_weights, strict=False)
        print(f"[load_qwen_pretrain_weight] load report (no embed/lm_head in ckpt): {err}")
        return model

    # 从 ckpt 读取 vocab 大小
    ckpt_vocab_size = None
    if ckpt_embed_key is not None:
        ckpt_vocab_size = renamed_weights[ckpt_embed_key].shape[0]
    elif ckpt_lm_head_key is not None:
        ckpt_vocab_size = renamed_weights[ckpt_lm_head_key].shape[0]

    # 当前模型 vocab 大小
    model_vocab_size = model.get_input_embeddings().weight.shape[0]

    if ckpt_vocab_size is None:
        # 理论不会走到这儿，兜底
        err = model.load_state_dict(renamed_weights, strict=False)
        print(f"[load_qwen_pretrain_weight] load report (no vocab info): {err}")
        return model

    # 若不一致，先把模型临时 resize 到 ckpt 的 vocab 尺寸（这样 load_state_dict 不会 shape mismatch）
    if ckpt_vocab_size != model_vocab_size:
        print(f"[load_qwen_pretrain_weight] temp resize model vocab {model_vocab_size} -> {ckpt_vocab_size}")
        model.resize_token_embeddings(ckpt_vocab_size)

    # 4) 加载（允许缺 key，但不允许 shape mismatch；我们已经对齐了 vocab）
    err = model.load_state_dict(renamed_weights, strict=False)
    print(f"[load_qwen_pretrain_weight] load report: {err}")

    return model

@torch.no_grad()
def load_qwen_pretrain_weight_with_no_vocab_resize(model, pretrain_weight_path: str):
    """
    等价于训练脚本里的 QwenVlAct_Trainer.load_qwen_pretrain_weight（去掉日志依赖），
    用于把 Qwen 的预训练 safetensors 权重加载并做 MoE 的 key 重映射。
    """
    from safetensors.torch import load_file

    weight_files = sorted(
        [f for f in os.listdir(pretrain_weight_path) if f.endswith(".safetensors")]
    )
    merged_weights = {}
    for weight_file in weight_files:
        file_path = os.path.join(pretrain_weight_path, weight_file)
        weights = load_file(file_path)
        merged_weights.update(weights)

    renamed_weights = {}
    for key, value in merged_weights.items():
        # MLP -> MoE experts 重命名
        if key.startswith("model.layers") and "mlp." in key and getattr(model.config, "mlp_moe", False):
            layer_num = key.split(".layers.")[1].split(".mlp")[0]
            new_key = key.replace(f"layers.{layer_num}.mlp.", f"layers.{layer_num}.moe.experts.0.")
            renamed_weights[new_key] = value
        # Attention -> experts 重命名
        elif key.startswith("model.layers") and "self_attn." in key and getattr(model.config, "attention_moe", False):
            layer_num = key.split(".layers.")[1].split(".self_attn")[0]
            proj_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
            hit = False
            for proj in proj_types:
                if f".{proj}." in key or key.endswith(f".{proj}.weight") or key.endswith(f".{proj}.bias"):
                    new_key = key.replace(
                        f"layers.{layer_num}.self_attn.{proj}",
                        f"layers.{layer_num}.self_attn.{proj}_experts.0",
                    )
                    renamed_weights[new_key] = value
                    hit = True
                    break
            if not hit:
                renamed_weights[key] = value
        else:
            renamed_weights[key] = value

    err = model.load_state_dict(renamed_weights, strict=False)
    print(f"[load_qwen_pretrain_weight] load report: {err}")
    print(f"[load_qwen_pretrain_weight] loaded from: {pretrain_weight_path}")
    return model


def init_qwen2_5_model(config: dict, accelerator: "Accelerator" = None):
    """
    仅做“模型初始化与权重加载”，对齐训练脚本里的 qwen2_5 分支。
    返回: model, processor, special_token_ids(dict)

    必要 config 键：
      - qwen_vl_act_config_path: Qwen2_5_VL 配置路径（可为本地或HF路径）
      - pretrained_wallx_path:   预训练权重/processor 路径（包含 safetensors 与 tokenizer）
    可选：
      - model_type:              默认 "qwen2_5"
      - flow_loss_weight:        默认 1.0
      - use_fast_tokenizer:      bool，默认 False
      - action_tokenizer_path:   当 use_fast_tokenizer=True 时需要
    """
    model_type = config.get("model_type", "qwen2_5")
    assert model_type in ["qwen2_5", "wall-oss"], f"Unsupported model_type: {model_type}"
    assert "qwen_vl_act_config_path" in config, "Missing qwen_vl_act_config_path"
    assert "pretrained_wallx_path" in config, "Missing pretrained_wallx_path"

    use_fast_tokenizer = config.get("use_fast_tokenizer", False)
    flow_loss_weight = config.get("flow_loss_weight", 1.0)

    # 1) 加载配置与 processor（训练脚本里对齐）
    qwen_cfg = Qwen2_5_VLConfig.from_pretrained(config["qwen_vl_act_config_path"])
    processor = AutoProcessor.from_pretrained(config["pretrained_wallx_path"], use_fast=True)

    # 2) （可选）接入动作 tokenizer 并注入新 token（训练脚本里的 fast tokenizer 流程）
    if use_fast_tokenizer:
        assert "action_tokenizer_path" in config, "use_fast_tokenizer=True 但缺少 action_tokenizer_path"
        action_tokenizer = AutoProcessor.from_pretrained(
            config["action_tokenizer_path"], trust_remote_code=True
        )
        # 注入专用 token
        new_tokens = ["<|propri|>", "<|action|>"]
        new_tokens += [f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)]
        processor.tokenizer.add_tokens(new_tokens)

        # 记录动作 token 的起始 index 与 vocab size 到 tokenizer 的 init_kwargs（与训练脚本一致）
        begin_idx_token = "<|action_token_0|>"
        token_id = processor.tokenizer.convert_tokens_to_ids(begin_idx_token)
        processor.tokenizer.init_kwargs["action_token_start_index"] = token_id
        processor.tokenizer.init_kwargs["action_token_vocab_size"] = action_tokenizer.vocab_size

        # 作为子处理器挂载，供后续动作编码使用
        processor.action_processor = action_tokenizer

    # 3) 构建模型（与训练脚本等价）
    model = Qwen2_5_VLMoEForAction(
        qwen_cfg,
        use_fast_tokenizer,
        processor,
        flow_loss_weight=flow_loss_weight,
    )

    # 4) 精度&加载预训练权重（顺序与训练脚本一致）
    model = model.to(torch.bfloat16)
    # model = load_qwen_pretrain_weight(model, config["pretrained_wallx_path"])
    model = load_qwen_pretrain_weight_with_no_vocab_resize(model, config["pretrained_wallx_path"])

    model.resize_token_embeddings(len(processor.tokenizer))
    model = model.to(torch.bfloat16)

    # 5) 设备放置（开环检测通常只需要推理）
    if accelerator is not None:
        model = accelerator.prepare(model)
        device = accelerator.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    model.eval()

    # 6) 返回常用特殊 token id，便于你后续拼接输入
    special_token_ids = {
        "propri_token_id": processor.tokenizer.convert_tokens_to_ids("<|propri|>"),
        "action_token_id": processor.tokenizer.convert_tokens_to_ids("<|action|>"),
    }

    print(f"[init_qwen2_5_model] device={device}, bf16={True}, fast_tokenizer={use_fast_tokenizer}")
    return model, processor, special_token_ids

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")
    return config

if __name__ == "__main__":
    origin_action_dim = 6
    pred_horizon = 5
    save_dir = "/data2/konghanlin/new_wallx/open_loop_figs"

    # 一个最简用例：你可以把路径替换成自己的
    example_config = {
        "model_type": "qwen2_5",
        "qwen_vl_act_config_path": "/data2/konghanlin/new_wallx/open_loop_figs/qwen25_config.json",
        "pretrained_wallx_path": "/data2/konghanlin/new_wallx/wallx_pt/Qwen2.5-VL-3B_noMOE/16/processor",  # 目录下应有 *.safetensors 与 tokenizer/processor
        "flow_loss_weight": 1.0,
        "use_fast_tokenizer": True,  # 若 False 可去掉 action_tokenizer_path
        "action_tokenizer_path": "/inspire/hdd/global_user/konghanlin-253108540238/fast_tokenizer",
    }

    acc = Accelerator() if Accelerator is not None else None

    model, processor, special_ids = init_qwen2_5_model(example_config, acc)
    print("loaded model!!!")
    model.eval()
    
    path = "/data2/konghanlin/new_wallx/workspace/lerobot_example/UAV_train/qwen2.5-3B-noMOE/config_qact.yml"
    config = load_config(path)

    
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})
    dataset = load_test_dataset(config, lerobot_config, seed=42)
    dataloader = dataset.get_dataloader()
    total_frames = len(dataloader)
    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 20 if predict_mode == "diffusion" else origin_action_dim
    gt_traj = torch.zeros((total_frames, origin_action_dim))
    pred_traj = torch.zeros((total_frames, origin_action_dim))
    for idx, batch in tqdm(
        enumerate(dataloader), total=total_frames, desc="predicting"
    ):
        if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
            batch = batch.to("cuda")
            with torch.no_grad():
                outputs = model(
                    **batch,
                    action_dim=action_dim,
                    pred_horizon=pred_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                )
                pred_traj[idx : idx + pred_horizon] = (
                    outputs["predict_action"][:, :, :origin_action_dim]
                    .detach()
                    .cpu()
                    .squeeze(0)
                )

                # 提取并解析 instruction 文本
                input_text = outputs["input_text"]
                
                # 若为 list，则拼接或取第一个元素
                if isinstance(input_text, list):
                    if len(input_text) > 0:
                        input_text = input_text[0]
                    else:
                        input_text = ""
                elif not isinstance(input_text, str):
                    input_text = str(input_text)  # 兜底转为字符串
                
                # 提取指令文本部分
                start_idx = input_text.find("Instruction:")
                mid_idx = input_text.find("You are performing a robotic manipulation task,")
                end_idx = input_text.find("If you believe the robot can now")

                if start_idx != -1 and mid_idx != -1:
                    instruction_part1 = input_text[start_idx + len("Instruction:"):mid_idx].strip()
                else:
                    instruction_part1 = ""

                if mid_idx != -1 and end_idx != -1:
                    instruction_part2 = input_text[mid_idx+len("You are performing a robotic manipulation task,"):end_idx].strip()
                else:
                    instruction_part2 = ""

                instruction_text = instruction_part1 + "\n" + instruction_part2            
                gt_action_chunk = batch["action_chunk"][:, :, :origin_action_dim]
            dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
            denormalized_gt = (
                model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [lerobot_config.get("repo_id", "physical-intelligence/libero")],
                    dof_mask,
                ).squeeze(0)
            )
            gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()

    gt_traj_np = gt_traj.numpy()
    pred_traj_np = pred_traj.numpy()

    timesteps = gt_traj.shape[0]


    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)  # 提高 dpi 提升画质

    # 绘制 Ground Truth 轨迹（蓝色）
    # ax.plot(gt_traj_np[:end, 0], gt_traj_np[:end, 1], label="Ground Truth", color="blue", marker='o', markersize=2, linewidth=1)

    # 绘制 Prediction 轨迹（红色）
    ax.plot(pred_traj_np[:, 0], pred_traj_np[:, 1], label="Prediction", color="red", marker='x', markersize=2, linewidth=1)


# 添加文本说明
    if instruction_text:
        fig.text(0.02, 0.02, instruction_text, fontsize=6, color='black', wrap=True,
                ha='left', va='bottom', transform=fig.transFigure)
    # 标注每个点编号（数字小一些）
    for p in range(len(gt_traj_np[:])):
        # ax.text(gt_traj_np[p, 0], gt_traj_np[p, 1], str(p), fontsize=10, color='blue', va='bottom', ha='right')
        ax.text(pred_traj_np[p, 0], pred_traj_np[p, 1], str(p), fontsize=10, color='red', va='bottom', ha='left')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectory Comparison')

    # 添加图例（去掉网格）
    ax.legend()
    ax.grid(False)

    # 保存高质量图像
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trajectory_test_episode_0.png"), dpi=300)  # 输出更高分辨率
    plt.close()