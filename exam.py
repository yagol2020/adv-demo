import numpy
from torchvision import datasets, transforms
from torchvision import models
import PIL
from torch.utils.data import DataLoader
import torch
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image,
    deprocess_image,
)
import numpy as np
from PIL import Image
import cv2
from utils import get_indexed_matrix, noise_it, noise_lz, random_noise_it

import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if os.path.exists("temp"):
    shutil.rmtree("temp")
os.mkdir("temp")

PLOT = True

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
batch_size = 20
noise_k_list = [i for i in range(1, 25)]
max_result_size = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.eval()
resnet50.to(device)


my_transforms = transforms.Compose(
    [
        transforms.Resize(256),  # 缩放到256*256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name, train, download, transform=None):
        self.dataset_name = dataset_name
        if dataset_name == "cifar100":
            self.dataset = datasets.CIFAR100(
                root=root, train=train, download=download, transform=None
            )
            self.classes = self.dataset.classes
        if dataset_name == "my":
            self.dataset = datasets.ImageFolder(root, transform=None)
            self.classes = self.dataset.classes
        self.transform = transform

    def __getitem__(self, index):
        # 获取原始图像和标签
        original_img, label = self.dataset[index]

        # 应用变换（如果有）
        if self.transform is not None:
            transformed_img = self.transform(original_img)
        else:
            transformed_img = original_img

        # 统一格式: PIL.Image.Image -> numpy.ndarray
        if isinstance(original_img, PIL.Image.Image):
            original_img = np.array(original_img)

        # 返回原始图像、变换后的图像和标签
        return (original_img, transformed_img), label

    def __len__(self):
        return len(self.dataset)


data_loader = DataLoader(
    MyDataSet(
        root="data/cifar100",
        dataset_name="cifar100",
        train=True,
        download=True,
        transform=my_transforms,
    ),
    batch_size=batch_size,
    shuffle=True,
)


def decode_predictions(output):
    """
    根据class id, 返回对应的类别名称
    """
    with open("imageNet_class.json") as f:
        labels = {int(key): value for key, value in json.load(f).items()}
    return labels[output.item()][1]


result = pd.DataFrame(
    columns=[
        "index",
        "noise_k",
        "label",
        "pred_origin",
        "pred_my",
        "pred_random",
        "pred_lz",
        "is_equal_my",
        "is_equal_random",
        "is_equal_lz",
    ]
)


for n_k in noise_k_list:
    for batch_index, (inputs, labels) in enumerate(data_loader):
        print(f"第{batch_index}批")
        input_tensor = inputs[1].to(device)
        model_output = resnet50(input_tensor)
        pred_class = model_output.argmax(dim=1)  # e.g., 501

        noise_my_batch = []  # 我的策略
        noise_random_batch = []  # 随机扰动策略
        noise_lz_batch = []  # lz的策略

        origin_pred_str_list = []  # 原始图像的预测结果, 可读字符串形式
        origin_imgs = []  # 原始图像, 彩色图像, 用于可视化比较
        heatmaps = []  # 热图, 用于可视化比较

        noise_my_imgs = []  # 我的策略生成的图像, 用于可视化比较
        noise_random_imgs = []  # 随机扰动策略生成的图像, 用于可视化比较
        noise_lz_imgs = []  # lz的策略生成的图像, 用于可视化比较

        for index in range(input_tensor.size(0)):
            predictions = decode_predictions(
                pred_class[index]
            )  # english class name, e.g., "English foxhound"
            origin_pred_str_list.append(predictions)
            img = inputs[0][index].squeeze(0).numpy()

            img = cv2.resize(img, (224, 224))
            origin_imgs.append(img)
            img_255 = np.float32(img) / 255

            # 用grad-cam生成热图矩阵
            with GradCAM(model=resnet50, target_layers=resnet50.layer4) as grad_cam:
                m_cam = grad_cam(
                    input_tensor, targets=[ClassifierOutputTarget(pred_class[index])]
                )
                img_with_heatmap = show_cam_on_image(img_255, m_cam[0, :], use_rgb=True)
                heatmaps.append(img_with_heatmap)

            indexed_m_cam = get_indexed_matrix(m_cam[0, :])
            # 我们的策略
            new_img = noise_it(img.copy(), indexed_m_cam, _top_k=n_k)
            noise_my_imgs.append(new_img)
            # 随机扰动策略
            new_img_random = random_noise_it(img.copy(), _top_k=n_k)
            noise_random_imgs.append(new_img_random)
            # lz的策略
            new_img_lz = noise_lz(img.copy(), indexed_m_cam, _top_k=n_k)
            noise_lz_imgs.append(new_img_lz)

            # 使用my_transforms对new_img进行预处理
            noise_input_tensor = my_transforms(PIL.Image.fromarray(new_img))
            noise_input_tensor = noise_input_tensor.to(device)
            noise_input_tensor = noise_input_tensor.unsqueeze(0)
            noise_my_batch.append(noise_input_tensor)

            # 使用my_transforms对new_img_random进行预处理
            noise_input_tensor_random = my_transforms(
                PIL.Image.fromarray(new_img_random)
            )
            noise_input_tensor_random = noise_input_tensor_random.to(device)
            noise_input_tensor_random = noise_input_tensor_random.unsqueeze(0)
            noise_random_batch.append(noise_input_tensor_random)

            # 使用my_transforms对new_img_lz进行预处理
            noise_input_tensor_lz = my_transforms(PIL.Image.fromarray(new_img_lz))
            noise_input_tensor_lz = noise_input_tensor_lz.to(device)
            noise_input_tensor_lz = noise_input_tensor_lz.unsqueeze(0)
            noise_lz_batch.append(noise_input_tensor_lz)

        noise_my_batch = torch.cat(noise_my_batch, dim=0)
        new_output = resnet50(noise_my_batch)
        new_y_pred = new_output.argmax(dim=1)

        noise_random_batch = torch.cat(noise_random_batch, dim=0)
        new_output_random = resnet50(noise_random_batch)
        new_y_pred_random = new_output_random.argmax(dim=1)

        noise_lz_batch = torch.cat(noise_lz_batch, dim=0)
        new_output_lz = resnet50(noise_lz_batch)
        new_y_pred_lz = new_output_lz.argmax(dim=1)

        predictions_new_list = []
        predictions_new_list_random = []
        predictions_new_list_lz = []

        for index in range(batch_size):
            noise_pred_english = decode_predictions(new_y_pred[index])

            noise_pred_english_random = decode_predictions(new_y_pred_random[index])

            noise_pred_english_lz = decode_predictions(new_y_pred_lz[index])

            predictions_new_list.append(noise_pred_english)

            predictions_new_list_random.append(noise_pred_english_random)

            predictions_new_list_lz.append(noise_pred_english_lz)

            result = pd.concat(
                [
                    result,
                    pd.DataFrame(
                        [
                            [
                                f"{batch_index}-{index}",
                                n_k,
                                data_loader.dataset.classes[labels[index]],
                                origin_pred_str_list[index],
                                predictions_new_list[index],
                                predictions_new_list_random[index],
                                predictions_new_list_lz[index],
                                origin_pred_str_list[index]
                                == predictions_new_list[index],
                                origin_pred_str_list[index]
                                == predictions_new_list_random[index],
                                origin_pred_str_list[index]
                                == predictions_new_list_lz[index],
                            ]
                        ],
                        columns=[
                            "index",
                            "noise_k",
                            "label",
                            "pred_origin",
                            "pred_my",
                            "pred_random",
                            "pred_lz",
                            "is_equal_my",
                            "is_equal_random",
                            "is_equal_lz",
                        ],
                    ),
                ]
            )

        if PLOT:
            for index in range(input_tensor.size(0)):
                if (
                    origin_pred_str_list[index] == predictions_new_list[index]
                    and origin_pred_str_list[index]
                    == predictions_new_list_random[index]
                    and origin_pred_str_list[index] == predictions_new_list_lz[index]
                ):  # 所有策略都对抗失败了, 不绘制
                    continue

                # 将img, img_with_heatmap, new_img拼接为一张图, 并在每幅图下面写上预测结果
                combined_img = cv2.hconcat(
                    [
                        origin_imgs[index],
                        heatmaps[index],
                        noise_my_imgs[index],
                        noise_lz_imgs[index],
                        noise_random_imgs[index],
                    ]
                )

                # 扩展图像以添加文本
                text_height = 30  # 为文本预留的高度

                extended_img = cv2.copyMakeBorder(
                    combined_img,
                    0,
                    text_height,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )

                # 定义文本和位置
                texts = [
                    origin_pred_str_list[index]
                    + "("
                    + data_loader.dataset.classes[labels[index]]
                    + ")",
                    "Heatmap",
                    predictions_new_list[index] + "(my)",
                    predictions_new_list_lz[index] + "(lz)",
                    predictions_new_list_random[index] + "(random)",
                ]
                # 图片的宽度
                img_width = combined_img.shape[1]
                # 单个图片的宽度
                single_img_width = img_width // 5
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5  # 字体大小
                color = (255, 255, 255)  # 白色文本
                thickness = 1

                # 在每个图像部分的下方添加文本
                for j, text in enumerate(texts):
                    # 计算文本的x坐标（水平位置）
                    text_x = (j * single_img_width) + (
                        single_img_width
                        - cv2.getTextSize(text, font, font_scale, thickness)[0][0]
                    ) // 2
                    # 计算文本的y坐标（垂直位置）
                    text_y = combined_img.shape[0] + text_height - 10  # 略高于底部
                    cv2.putText(
                        extended_img,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )

                cv2.imwrite(f"temp/{n_k}-{batch_index}.jpg", extended_img)
                break

        if batch_index * batch_size >= max_result_size:
            break

result.to_csv("result.csv", index=False)

# 分析结果
result = pd.read_csv("result.csv")
# 绘制折线图, 每条折线代表一种策略, x轴为noise_k, y轴为对抗成功率

success_rate_my = (
    result[result["is_equal_my"] == False].groupby("noise_k").count()["index"]
    / result.groupby("noise_k").count()["index"]
)
success_rate_random = (
    result[result["is_equal_random"] == False].groupby("noise_k").count()["index"]
    / result.groupby("noise_k").count()["index"]
)
success_rate_lz = (
    result[result["is_equal_lz"] == False].groupby("noise_k").count()["index"]
    / result.groupby("noise_k").count()["index"]
)

print(success_rate_my)
print(success_rate_random)
print(success_rate_lz)

sns.lineplot(x=success_rate_my.index, y=success_rate_my.values, label="my")
sns.lineplot(x=success_rate_random.index, y=success_rate_random.values, label="random")
sns.lineplot(x=success_rate_lz.index, y=success_rate_lz.values, label="lz")
plt.xlabel("noise_k")
plt.ylabel("success_rate")
plt.xticks(success_rate_my.index)
plt.legend()
plt.tight_layout()
plt.savefig("success_rate.pdf")
