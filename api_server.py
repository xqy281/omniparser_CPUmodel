# api_server.py

import asyncio
import base64
import io
from functools import partial
from typing import Dict, Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# 从项目原有工具脚本中导入核心功能
from util.utils import (check_ocr_box, get_caption_model_processor,
                        get_som_labeled_img, get_yolo_model, predict_yolo,
                        remove_overlap_new, get_parsed_content_icon, int_box_area)

# --- 1. 全局初始化 ---
print("正在加载模型，请稍候...")
DEVICE = torch.device('cpu')

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt').to(DEVICE)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
    device=DEVICE
)
print("模型加载完成！服务准备就绪。")


# --- 2. 核心解析逻辑 (保留了原始的带图生成功能) ---
def run_omniparser_processing_with_image(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int
) -> Dict[str, Any]:
    """
    原始的处理流程，生成并返回带标注的图片。
    """
    try:
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        ocr_bbox_rslt, _ = check_ocr_box(
            image_input, display_img=False, output_bb_format='xyxy', use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        annotated_image_b64, _, parsed_content_list = get_som_labeled_img(
            image_input, yolo_model, BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True, ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor,
            ocr_text=text, iou_threshold=iou_threshold, imgsz=imgsz,
        )
        return {
            "status": "success",
            "parsed_elements": parsed_content_list,
            "annotated_image_base64": annotated_image_b64
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# --- 3. 新增：高效的无图解析逻辑 ---
def run_omniparser_processing_headless(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int
) -> Dict[str, Any]:
    """
    优化的无图处理流程，跳过所有图像绘制和编码步骤，只返回结构化数据。
    """
    try:
        w, h = image_input.size
        image_source_np = np.asarray(image_input)

        # 步骤 1: OCR
        ocr_bbox_rslt, _ = check_ocr_box(
            image_input, display_img=False, output_bb_format='xyxy', use_paddleocr=use_paddleocr
        )
        ocr_text, ocr_bbox_pixel = ocr_bbox_rslt

        # 步骤 2: YOLO 检测
        xyxy_pixel, _, _ = predict_yolo(model=yolo_model, image=image_input, box_threshold=box_threshold, imgsz=imgsz, scale_img=False)
        
        # 坐标归一化
        ocr_bbox_ratio = (torch.tensor(ocr_bbox_pixel) / torch.Tensor([w, h, w, h])).tolist() if ocr_bbox_pixel else []
        xyxy_ratio = (xyxy_pixel / torch.Tensor([w, h, w, h])).tolist()

        # 步骤 3: 整合与去重
        ocr_bbox_elem = [{'type': 'text', 'bbox': box, 'interactivity': False, 'content': txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(ocr_bbox_ratio, ocr_text) if int_box_area(box, w, h) > 0]
        xyxy_elem = [{'type': 'icon', 'bbox': box, 'interactivity': True, 'content': None} for box in xyxy_ratio if int_box_area(box, w, h) > 0]
        filtered_boxes_elem = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
        
        filtered_boxes_sorted = sorted(filtered_boxes_elem, key=lambda x: x['content'] is None)
        starting_idx = next((i for i, box in enumerate(filtered_boxes_sorted) if box['content'] is None), -1)
        filtered_boxes_tensor = torch.tensor([box['bbox'] for box in filtered_boxes_sorted])

        # 步骤 4: 图像描述 (最耗时的部分)
        parsed_content_icon = get_parsed_content_icon(filtered_boxes_tensor, starting_idx, image_source_np, caption_model_processor)
        
        # 填充描述内容
        icon_content_iterator = iter(parsed_content_icon)
        for elem in filtered_boxes_sorted:
            if elem['content'] is None:
                try:
                    elem['content'] = next(icon_content_iterator)
                except StopIteration:
                    break
        
        return {
            "status": "success",
            "parsed_elements": filtered_boxes_sorted
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# --- 4. 创建 FastAPI 应用 ---
app = FastAPI(
    title="OmniParser API",
    description="一个用于将 GUI 屏幕截图解析为结构化元素的服务。",
    version="1.1.0"
)


# --- 5. 修正后的 API 端点 ---
@app.post("/parse_image/")
async def parse_image_endpoint(
    image: UploadFile = File(..., description="要解析的图像文件"),
    box_threshold: float = Form(0.05, description="检测框的置信度阈值"),
    iou_threshold: float = Form(0.1, description="用于NMS的IOU阈值"),
    use_paddleocr: bool = Form(True, description="是否使用PaddleOCR"),
    imgsz: int = Form(640, description="图标检测的图像尺寸"),
    generate_annotated_image: bool = Form(True, description="是否生成并返回带标注的图片")
):
    try:
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"无法读取或解析图片: {e}"})

    loop = asyncio.get_event_loop()

    # 根据标志选择不同的处理函数
    if generate_annotated_image:
        func_to_run = partial(
            run_omniparser_processing_with_image,
            image_input=input_image, box_threshold=box_threshold,
            iou_threshold=iou_threshold, use_paddleocr=use_paddleocr, imgsz=imgsz
        )
    else:
        func_to_run = partial(
            run_omniparser_processing_headless,
            image_input=input_image, box_threshold=box_threshold,
            iou_threshold=iou_threshold, use_paddleocr=use_paddleocr, imgsz=imgsz
        )

    result = await loop.run_in_executor(None, func_to_run)

    if result["status"] == "success":
        return JSONResponse(status_code=200, content=result)
    else:
        return JSONResponse(status_code=500, content=result)


# --- 6. 启动服务器 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)