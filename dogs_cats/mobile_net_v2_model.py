import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# 加载预训练的 MobileNetV2 模型
model = MobileNetV2(weights='imagenet')

# 定义一个图片分类函数
def classify_image(image_path):
    # 加载图片并调整到模型需要的尺寸 (224x224)
    img = load_img(image_path, target_size=(224, 224))
    #plt.imshow(img)  # 显示图片
    #plt.axis('off')
    #plt.show()

    # 将图片转换为模型需要的数组格式
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    img_array = preprocess_input(img_array)  # 对图片进行预处理

    # 预测图片类别
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)  # 解码预测结果（取前3名）

    # 打印预测结果
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} - {score:.4f}")

# 测试分类函数（使用你自己的图片路径）
image_path = "/Users/wangdi/Downloads/kagglecatsanddogs_3367a/PetImages/Dog/0.jpg"  # 替换为实际图片路径
classify_image(image_path)
