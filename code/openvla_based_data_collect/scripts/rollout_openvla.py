from PIL import Image
import keyboard

import numpy as np
import cv2
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


from real_env import make_real_env
from convert_ee import wx250s, ModifiedIKinSpace, get_joint_and_gripper, get_ee

def get_image(camera_port=0):

    # 打开指定端口的摄像头
    cap = cv2.VideoCapture(camera_port)

    # 检查摄像头是否成功打开
    while True:
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_port}")
            # return None, False
        else:
            break

    # # 读取一帧图像
    ret, frame = cap.read()

    # if frame is not None:
    #     cv2.imshow("Camera Image", frame)
    #     cv2.waitKey(0)  # 等待按键
    #     cv2.destroyAllWindows()  # 关闭所有窗口
    # else:
    #     print("没有图像可以显示")

    # 释放摄像头资源
    cap.release()

    if not ret:
        print("无法读取图像")
        return None, False

    return Image.fromarray(frame)


start_matrix = np.array(
    [
        [0.267, 0.000, 0.963, float(0.3)],
        [0.000, 1.000, 0.000, float(-0.09)],
        [-0.963, 0.000, 0.267, float(0.26)],
        [0.00, 0.00, 0.00, 1.00],
    ]
)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:1")

# 初始化机械臂
env = make_real_env(init_node=True)
# bring arm to start_position
env.reset()

# guess_angle = np.array([-0.6166603, -0.10124274, 0.02914564, 0.0076699, 1.40972841, -0.21935926])
guess_angle = np.array([-0.33609906, -0.08757106, -0.10869286, 0.08838999, 1.51128642, -0.32999209])

start_joint_position, success = ModifiedIKinSpace(wx250s.Slist, wx250s.M, start_matrix, guess_angle, eomg=1e-3, ev=1e-4)
# print(start_joint_position)


start_ee_position = get_ee(start_joint_position, gripper=np.array([0.5]))
action = start_ee_position

while True:
    # Grab image input & format prompt
    image = get_image(camera_port=0)
    # prompt = "In: What action should the robot take to {lift Purple cup}?\nOut:"
    prompt = "In: What action should the robot take to {lift Red cup}?\nOut:"
    # prompt = "In: What action should the robot take to {Put Cup from Counter into Sink}?\nOut:"
    # prompt = "In: What action should the robot take to {Put Purple Cup on White Plate}?\nOut:"
    # prompt = "In: What action should the robot take to {Put Eggplant into Pot}?\nOut:"
    # prompt = "In: What action should the robot take to {Put Lid on Pot}?\nOut:"
    # prompt = "In: What action should the robot take to {Stack Green Cup on Purple Cup}?\nOut:"

    # prompt = "In: What action should the robot take to {Flip Cup Upright}?\nOut:"
    # prompt = "In: What action should the robot take to {Flip Pot Upright}?\nOut:"
    # prompt = "In: What action should the robot take to {Put Carrot on Plate}?\nOut:" # not
    # prompt = "In: What action should the robot take to {Take Eggplant out of Pot}?\nOut:"

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to("cuda:1", dtype=torch.bfloat16)
    delta_action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    action = (action + delta_action).squeeze()

    action_joint_position = get_joint_and_gripper(action[:3], action[3:6], delta_action[6:7], guess_angle)
    guess_angle = action_joint_position[:6]
    # print(action_joint_position[-1])

    env.step(action_joint_position)


print('arrive start_position')
env.reset()
