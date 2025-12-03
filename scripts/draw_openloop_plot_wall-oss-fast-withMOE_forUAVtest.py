import os
import sys
sys.path.insert(0, "/mnt/diff-ali/workspace/wall-x")

import yaml
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_horizon", type=int, default=5)
    parser.add_argument("--origin_action_dim", type=int, default=6)
    args = parser.parse_args()

    origin_action_dim = args.origin_action_dim
    pred_horizon = args.pred_horizon

    model_path = "/mnt/diff-ali/workspace/wall-x/wallx_pt/walloss-fast-moe-for-sim/processor"
    action_tokenizer_path = "/mnt/diff-ali/workspace/wall-x/fast-tokenizer"
    save_dir = "/mnt/diff-ali/workspace/wall-x/open_loop_figs"
    path = "/mnt/diff-ali/workspace/wall-x/workspace/lerobot_example/UAV_test/wall-oss_fast-withMOE/config_qact.yml"
    config = load_config(path)

    # load model with customized robot config
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        pretrained_model_path=model_path, train_config=config, action_tokenizer_path=action_tokenizer_path
    )
    model.eval().to("cuda").bfloat16()

    # get test dataloader
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})
    dataset = load_test_dataset(config, lerobot_config, episode = list(range(50)),seed=42)
    dataloader = dataset.get_dataloader()

    total_frames = len(dataloader)
    episode_index = 0
    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 20 if predict_mode == "diffusion" else origin_action_dim
    gt_traj = torch.zeros((total_frames, origin_action_dim))
    pred_traj = torch.zeros((total_frames, origin_action_dim))

    step = 0#表示循环中，现在处在第episode_index个episode中的第step步
    pic = 0
    begin_plot = 0#从第几步开始绘制
    CONST_ZERO_TENSOR = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    
    for idx, batch in tqdm(
        enumerate(dataloader), total=total_frames, desc="predicting"
    ):
        if (torch.equal(batch.data['action_chunk'][0][0], batch.data['action_chunk'][0][1]) and step!=0) or (idx == total_frames-1):
            if idx != 0:
                gt_traj_np = gt_traj.numpy()
                pred_traj_np = pred_traj.numpy()

                # ==================== 修改绘图部分 ====================
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"wall-oss-fast-withMOE_{pic}.png")
                pic += 1
                fig = plt.figure(figsize=(10, 10))
                plt.title("XY Trajectory Comparison (lerobot)")

                # Ground truth trajectory
                plt.plot(gt_traj_np[begin_plot:step, 0], gt_traj_np[begin_plot:step, 1], '-o', label='Ground Truth', alpha=0.7)
                # Predicted trajectory
                plt.plot(pred_traj_np[begin_plot:step, 0], pred_traj_np[begin_plot:step, 1], '-o', label='Prediction', alpha=0.7)

                # 标出编号（step index）
                for i in range(begin_plot, step):
                    plt.text(gt_traj_np[i, 0], gt_traj_np[i, 1], str(i), fontsize=8, color='blue')
                    plt.text(pred_traj_np[i, 0], pred_traj_np[i, 1], str(i), fontsize=8, color='orange')

                plt.xlabel("Action Dim 1 (X)")
                plt.ylabel("Action Dim 2 (Y)")
                plt.legend()
                plt.axis('equal')
                plt.tight_layout()

                
                if instruction_text:
                    # 使用 fig.text 而非 plt.text，因为要在整张图坐标系中添加文本
                    fig.text(0.02, 0.02, instruction_text, fontsize=6, color='black', wrap=True,
                            ha='left', va='bottom')
                # 高分辨率保存
                plt.savefig(save_path, dpi=600, bbox_inches='tight')
                plt.close()
                print(f"Saved high-resolution XY trajectory plot to {save_path}")
                gt_traj = torch.zeros((total_frames, origin_action_dim))
                pred_traj = torch.zeros((total_frames, origin_action_dim))
                step = 0
                continue
        if step % pred_horizon == 0 and step + pred_horizon < total_frames and step >=0:
            if step == 0 and abs(batch.data['action_chunk'][0][0][0])+abs(batch.data['action_chunk'][0][0][1])>0.05:
                continue
            batch = batch.to("cuda")
            with torch.no_grad():
                outputs = model(
                    **batch,
                    action_dim=action_dim,
                    pred_horizon=pred_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                )
                try: 
                    pred_traj[step : step + pred_horizon] = (
                        outputs["predict_action"][:, :, :origin_action_dim]
                        .detach()
                        .cpu()
                        .squeeze(0)
                    )
                except :
                    continue
                
                
                
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
            # gt_traj[step : step + pred_horizon] = batch["action_chunk"][:, :pred_horizon, :origin_action_dim]
            gt_action_chunk = batch["action_chunk"][:, :, :origin_action_dim]
            dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
            denormalized_gt = (
                model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [lerobot_config.get("repo_id", "physical-intelligence/libero")],
                    dof_mask,
                ).squeeze(0)
            )
            gt_traj[step : step + pred_horizon] = denormalized_gt.detach().cpu()
            
        
            # if torch.equal(batch.data['action_chunk'][0][-1], batch.data['action_chunk'][0][-2]):
            #     print(f"\n第{episode_index}个episode过了!\nidx:",idx)
            #     episode_index += 1
            #     for i in range(2, len(batch.data['action_chunk'][0]) + 1):  # 从倒数第二个开始
            #         if not torch.equal(batch.data['action_chunk'][0][-i], batch.data['action_chunk'][0][-1]):
            #             print(f"停在未来第 {len(batch.data['action_chunk'][0])-i+2} 个动作")
            #             step = -(len(batch.data['action_chunk'][0])-i)
            #             break
                

        step += 1
