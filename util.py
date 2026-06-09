# import cv2
# import os
# from pathlib import Path
#
#
# def parse_coordinates(coord_str):
#     """解析坐标字符串，返回(x_min, y_min, x_max, y_max)"""
#     coords = list(map(int, coord_str.strip().split(',')))
#     # 提取所有x和y坐标
#     x_coords = coords[0::2]  # 索引0,2,4,6
#     y_coords = coords[1::2]  # 索引1,3,5,7
#
#     # 获取最小和最大值
#     x_min, x_max = min(x_coords), max(x_coords)
#     y_min, y_max = min(y_coords), max(y_coords)
#
#     return x_min, y_min, x_max, y_max
#
#
# def sort_characters(characters):
#     """
#     对字符进行排序：从右往左（x坐标降序），从上往下（y坐标升序）
#     每个字符格式：(x_min, y_min, x_max, y_max, original_coord)
#     """
#     # 先按行分组（基于y坐标的近似值）
#     # 获取所有y_min值并排序
#     y_values = sorted(set([char[1] for char in characters]))
#
#     # 将相近的y坐标分为同一行（允许有小的误差）
#     rows = []
#     y_threshold = 10  # 同一行的y坐标差异阈值
#
#     for char in characters:
#         placed = False
#         for row in rows:
#             # 检查当前字符是否属于已有行
#             if abs(char[1] - row['y_center']) <= y_threshold:
#                 row['chars'].append(char)
#                 placed = True
#                 break
#
#         if not placed:
#             # 创建新行
#             rows.append({
#                 'y_center': char[1],
#                 'chars': [char]
#             })
#
#     # 对每一行内的字符按x坐标降序排序（从右往左）
#     for row in rows:
#         row['chars'].sort(key=lambda c: c[2], reverse=True)  # 使用x_max排序
#
#     # 按y坐标升序排序行（从上往下）
#     rows.sort(key=lambda r: r['y_center'])
#
#     # 合并所有行
#     sorted_characters = []
#     for row in rows:
#         sorted_characters.extend(row['chars'])
#
#     return sorted_characters
#
#
# def read_coords_from_file(file_path):
#     """从txt文件读取坐标数据"""
#     coords_list = []
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if line:  # 跳过空行
#                     coords_list.append(line)
#         print(f"成功读取 {len(coords_list)} 个坐标数据")
#         return coords_list
#     except Exception as e:
#         print(f"读取文件失败: {e}")
#         return []
#
#
# def crop_and_save(image_path, coords_list, output_folder, prefix='char', margin=2):
#     """
#     根据坐标列表裁剪图片并保存
#
#     Args:
#         image_path: 原始图片路径
#         coords_list: 坐标列表，每个元素是坐标字符串
#         output_folder: 输出文件夹路径
#         prefix: 输出文件前缀
#         margin: 裁剪边距
#     """
#     # 读取图片
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"无法读取图片: {image_path}")
#         return False
#
#     print(f"图片尺寸: {img.shape[1]} x {img.shape[0]}")
#
#     # 创建输出文件夹
#     Path(output_folder).mkdir(parents=True, exist_ok=True)
#
#     # 解析所有坐标
#     characters = []
#     for coord_str in coords_list:
#         x_min, y_min, x_max, y_max = parse_coordinates(coord_str)
#
#         # 添加边距并确保不超出图片边界
#         x_min = max(0, x_min - margin)
#         y_min = max(0, y_min - margin)
#         x_max = min(img.shape[1], x_max + margin)
#         y_max = min(img.shape[0], y_max + margin)
#
#         # 确保裁剪区域有效
#         if x_min < x_max and y_min < y_max:
#             characters.append((x_min, y_min, x_max, y_max, coord_str))
#         else:
#             print(f"警告: 无效的坐标区域: {coord_str}")
#
#     if not characters:
#         print("没有有效的坐标数据")
#         return False
#
#     # 按指定顺序排序
#     sorted_chars = sort_characters(characters)
#
#     # 裁剪并保存
#     saved_count = 0
#     for idx, (x_min, y_min, x_max, y_max, original_coord) in enumerate(sorted_chars, 1):
#         try:
#             # 裁剪区域
#             crop = img[y_min:y_max, x_min:x_max]
#
#             # 检查裁剪区域是否为空
#             if crop.size == 0:
#                 print(f"警告: 裁剪区域为空: {original_coord}")
#                 continue
#
#             # 生成文件名（从001开始编号）
#             filename = f"{prefix}_{idx:03d}.png"
#             filepath = os.path.join(output_folder, filename)
#
#             # 保存图片
#             cv2.imwrite(filepath, crop)
#             print(f"已保存: {filename} - 坐标: ({x_min},{y_min},{x_max},{y_max})")
#             saved_count += 1
#
#         except Exception as e:
#             print(f"保存失败 {original_coord}: {e}")
#
#     print(f"\n处理完成！共保存了 {saved_count} 个字符图片到: {output_folder}")
#     return True
#
#
# def main():
#     """主函数"""
#     # 配置文件路径
#     image_path = "/homes/hwj/code/OCR/test/image/句容金石記_34.png"  # 请修改为你的图片路径
#     coords_file = "/homes/hwj/code/OCR/test/mask_word/句容金石記_34.txt"  # 请修改为你的坐标文件路径
#     output_folder = "/homes/hwj/code/OCR/test/image_patch_chars"  # 输出文件夹
#
#     # 检查文件是否存在
#     if not os.path.exists(image_path):
#         print(f"错误: 图片文件不存在 - {image_path}")
#         return
#
#     if not os.path.exists(coords_file):
#         print(f"错误: 坐标文件不存在 - {coords_file}")
#         return
#
#     # 读取坐标数据
#     coords_list = read_coords_from_file(coords_file)
#
#     if not coords_list:
#         print("没有读取到坐标数据")
#         return
#
#     # 执行裁剪和保存
#     crop_and_save(image_path, coords_list, output_folder, prefix="char", margin=2)
#
#     print("\n提示: 你可以调整以下参数来优化结果:")
#     print("  - margin: 裁剪边距（当前为2像素）")
#     print("  - y_threshold: 行合并阈值（当前为10像素）")
#
#
# if __name__ == "__main__":
#     main()


# import os
#
# def expand_box(coords, expand=5, img_w=None, img_h=None):
#     """
#     coords: [x1,y1,x2,y2,x3,y3,x4,y4]
#     expand: 扩张像素
#     """
#
#     xs = coords[0::2]
#     ys = coords[1::2]
#
#     xmin = min(xs) - expand
#     xmax = max(xs) + expand
#     ymin = min(ys) - expand
#     ymax = max(ys) + expand
#
#     # 防止越界（如果提供图像尺寸）
#     if img_w is not None:
#         xmin = max(0, xmin)
#         xmax = min(img_w - 1, xmax)
#     if img_h is not None:
#         ymin = max(0, ymin)
#         ymax = min(img_h - 1, ymax)
#
#     # 重新构造矩形（保持顺序一致）
#     new_coords = [
#         int(xmin), int(ymin),
#         int(xmax), int(ymin),
#         int(xmax), int(ymax),
#         int(xmin), int(ymax)
#     ]
#
#     return new_coords
#
#
# def process_txt_file(in_path, out_path, expand=5, img_w=None, img_h=None):
#     with open(in_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     new_lines = []
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#
#         coords = list(map(int, line.split(',')))
#         new_coords = expand_box(coords, expand, img_w, img_h)
#
#         new_lines.append(','.join(map(str, new_coords)))
#
#     with open(out_path, 'w', encoding='utf-8') as f:
#         f.write('\n'.join(new_lines))
#
#
# def batch_process(input_dir, output_dir, expand=5, img_w=None, img_h=None):
#     os.makedirs(output_dir, exist_ok=True)
#
#     for filename in os.listdir(input_dir):
#         if not filename.endswith('.txt'):
#             continue
#
#         in_path = os.path.join(input_dir, filename)
#         out_path = os.path.join(output_dir, filename)
#
#         process_txt_file(in_path, out_path, expand, img_w, img_h)
#
#     print("✅ 全部处理完成！")
#
#
# # ===== 使用示例 =====
# if __name__ == "__main__":
#     input_dir = "/homes/hwj/code/BasicSR-master/CRAFT/result_masks"
#     output_dir = "/homes/hwj/code/BasicSR-master/CRAFT/result_masks_1"
#
#     batch_process(input_dir, output_dir, expand=5)


# import os
# import cv2
# import numpy as np
#
#
# def crop_char(img, coords):
#     """
#     根据四边形坐标裁剪字符（支持旋转）
#     coords: [x1,y1,x2,y2,x3,y3,x4,y4]
#     """
#
#     pts = np.array(coords, dtype=np.float32).reshape(4, 2)
#
#     # 计算宽高
#     w = int(max(
#         np.linalg.norm(pts[0] - pts[1]),
#         np.linalg.norm(pts[2] - pts[3])
#     ))
#
#     h = int(max(
#         np.linalg.norm(pts[1] - pts[2]),
#         np.linalg.norm(pts[3] - pts[0])
#     ))
#
#     if w <= 0 or h <= 0:
#         return None
#
#     # 目标矩形
#     dst_pts = np.array([
#         [0, 0],
#         [w - 1, 0],
#         [w - 1, h - 1],
#         [0, h - 1]
#     ], dtype=np.float32)
#
#     # 透视变换
#     M = cv2.getPerspectiveTransform(pts, dst_pts)
#     crop = cv2.warpPerspective(img, M, (w, h))
#
#     return crop
#
#
# def find_image(image_dir, name):
#     exts = ['.jpg', '.png', '.jpeg', '.bmp']
#     for ext in exts:
#         path = os.path.join(image_dir, name + ext)
#         if os.path.exists(path):
#             return path
#     return None
#
#
# def process_one(txt_path, image_dir, output_dir):
#     name = os.path.splitext(os.path.basename(txt_path))[0]
#
#     img_path = find_image(image_dir, name)
#     if img_path is None:
#         print(f"⚠️ 找不到图像: {name}")
#         return
#
#     img = cv2.imread(img_path)
#
#     with open(txt_path, 'r', encoding='utf-8') as f:
#         lines = [l.strip() for l in f if l.strip()]
#
#     for idx, line in enumerate(lines):
#         coords = list(map(float, line.split(',')))
#
#         crop = crop_char(img, coords)
#         if crop is None:
#             continue
#
#         save_name = f"{name}_{idx+1:02d}.jpg"
#         save_path = os.path.join(output_dir, save_name)
#
#         cv2.imwrite(save_path, crop)
# def batch_crop(txt_dir, image_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#
#     for filename in os.listdir(txt_dir):
#         if not filename.endswith('.txt'):
#             continue
#
#         txt_path = os.path.join(txt_dir, filename)
#         process_one(txt_path, image_dir, output_dir)
#
#     print("✅ 字符裁剪完成！")
#
#
# # ===== 使用 =====
# if __name__ == "__main__":
#     txt_dir = "/homes/hwj/code/BasicSR-master/CRAFT/result_masks_1"
#     image_dir = "/homes/hwj/data/song/image_patch"
#     output_dir = "/homes/hwj/data/song/image_patch_char"
#
#     batch_crop(txt_dir, image_dir, output_dir)




