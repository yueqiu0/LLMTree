import cv2
import numpy as np
import os

def create_clean_montage(image_paths, output_path, base_height=900, text_scale=1.5):
    """
    纯净版图片拼接工具
    直接显示alpha值，无特殊字符，统一小数格式
    """
    # 路径验证
    print("正在验证图片路径...")
    missing_files = [p for p in image_paths if not os.path.exists(p)]
    if missing_files:
        print("\n缺失文件：")
        [print(f" - {os.path.abspath(p)}") for p in missing_files]
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 1. 加载图片
        print("\n加载图片中...")
        images = []
        max_width = 0
        for path, alpha in image_paths.items():
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"图片读取失败: {path}")
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            images.append((img, alpha))
            max_width = max(max_width, img.shape[1])
            print(f"已加载: {os.path.basename(path)} | 尺寸: {img.shape}")

        # 2. 按alpha排序
        images.sort(key=lambda x: x[1])

        # 3. 计算目标尺寸
        target_height = base_height
        target_width = int(target_height * 1.6)  # 固定1.6:1比例
        
        # 确保能容纳最宽图片
        if max_width > target_width:
            target_width = min(max_width, int(target_height * 2.5))
            print(f"调整宽度以容纳最宽图片: {target_width}px")

        print(f"目标尺寸: {target_width}x{target_height}")

        # 4. 图片标准化
        print("\n标准化处理图片...")
        processed = []
        for img, alpha in images:
            # 保持比例缩放
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            
            resized = cv2.resize(img, (new_w, target_height), 
                                interpolation=cv2.INTER_LANCZOS4)
            
            # 居中填充
            if new_w < target_width:
                delta = target_width - new_w
                left = delta // 2
                right = delta - left
                resized = cv2.copyMakeBorder(resized, 0, 0, left, right,
                                            cv2.BORDER_CONSTANT, value=[255,255,255])
            
            processed.append((resized, alpha))
            print(f"alpha={alpha}: {img.shape} -> {resized.shape}")

        # 5. 创建画布
        rows, cols = 2, 3
        h_pad, v_pad = 80, 80
        title_space = 150
        
        montage_w = cols * target_width + (cols + 1) * h_pad
        montage_h = rows * target_height + (rows + 1) * v_pad + title_space
        
        montage = np.ones((montage_h, montage_w, 3), dtype=np.uint8) * 255
        print(f"\n画布尺寸: {montage_w}x{montage_h}")

        # 6. 添加标题
        font = cv2.FONT_HERSHEY_DUPLEX
        title = "abalone quick Analysis"
        
        title_size = cv2.getTextSize(title, font, 1.8, 3)[0]
        cv2.putText(montage, title,
                   ((montage_w - title_size[0]) // 2, 120),
                   font, 1.8, (0, 0, 150), 3)

        # 7. 排列图片
        print("\n排列图片...")
        for i, (img, alpha) in enumerate(processed):
            row = i // cols
            col = i % cols
            
            x = h_pad + col * (target_width + h_pad)
            y = title_space + v_pad + row * (target_height + v_pad)
            
            # 安全放置检查
            if y + img.shape[0] > montage_h or x + img.shape[1] > montage_w:
                raise ValueError("图片超出画布边界")
            
            montage[y:y+img.shape[0], x:x+img.shape[1]] = img
            
            # 纯净版标注（直接显示alpha值）
            label = f"alpha = {alpha:.1f}".replace(".0", "")  # 去除.0
            (label_w, label_h), _ = cv2.getTextSize(label, font, text_scale, 2)
            
            # 增大标注区域（高度增加50%）
            bg_height = int(label_h * 1.5)
            cv2.rectangle(montage,
                         (x, y + target_height),
                         (x + target_width, y + target_height + bg_height + 30),
                         (240, 240, 240), -1)
            cv2.rectangle(montage,
                         (x, y + target_height),
                         (x + target_width, y + target_height + bg_height + 30),
                         (200, 200, 200), 1)
            
            # 居中文字
            cv2.putText(montage, label,
                       (x + (target_width - label_w) // 2, y + target_height + bg_height + 10),
                       font, text_scale, (0, 0, 200), 2)
            
            print(f" 位置 ({x},{y}) | {label}")

        # 8. 保存结果
        cv2.imwrite(output_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n完成！图片已保存到: {os.path.abspath(output_path)}")
        return True

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# 使用您的路径配置
IMAGE_PATHS = {
    'C:/Users/chenx/Downloads/viz/abalone_quick_16_0_improvements.png': 0,
    'C:/Users/chenx/Downloads/viz/abalone_quick_16_0dot2_improvements.png': 0.2,
    'C:/Users/chenx/Downloads/viz/abalone_quick_16_0dot4_improvements.png': 0.4,
    'C:/Users/chenx/Downloads/viz/abalone_quick_16_0dot6_improvements.png': 0.6,
    'C:/Users/chenx/Downloads/viz/abalone_quick_16_0dot8_improvements.png': 0.8,
    'C:/Users/chenx/Downloads/viz/abalone_quick_16_1_improvements.png': 1.0
}

OUTPUT_PATH = 'C:/Users/chenx/Downloads/abalone_quick_final_result.jpg'

# 执行
if __name__ == "__main__":
    print("===== 图片拼接开始 =====")
    success = create_clean_montage(
        image_paths=IMAGE_PATHS,
        output_path=OUTPUT_PATH,
        base_height=900,
        text_scale=1.5
    )
    
    if success:
        print("\n成功生成拼接图！")
        print(f"输出文件: {os.path.abspath(OUTPUT_PATH)}")
    else:
        print("\n生成失败，请检查：")
        print("1. 所有图片路径是否正确")
        print("2. 图片是否能正常打开")
        print("3. 输出目录是否有权限")