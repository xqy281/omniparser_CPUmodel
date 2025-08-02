# api_server.py (Integrated Version - Fixed)

import asyncio
import io
import base64  # <--- 关键修正：在这里导入 base64 库
from functools import partial
from typing import Dict, Any

import gradio as gr
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

# --- 1. 全局初始化 (只执行一次) ---
print("正在加载共享模型，请稍候...")
DEVICE = torch.device('cpu')

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt').to(DEVICE)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
    device=DEVICE
)
print("模型加载完成！服务准备就绪。")


# --- 2. API 核心逻辑 (保持不变) ---
def run_omniparser_processing_with_image(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int
) -> Dict[str, Any]:
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

def run_omniparser_processing_headless(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int
) -> Dict[str, Any]:
    try:
        w, h = image_input.size
        image_source_np = np.asarray(image_input)
        ocr_bbox_rslt, _ = check_ocr_box(
            image_input, display_img=False, output_bb_format='xyxy', use_paddleocr=use_paddleocr
        )
        ocr_text, ocr_bbox_pixel = ocr_bbox_rslt
        xyxy_pixel, _, _ = predict_yolo(model=yolo_model, image=image_input, box_threshold=box_threshold, imgsz=imgsz, scale_img=False)
        ocr_bbox_ratio = (torch.tensor(ocr_bbox_pixel) / torch.Tensor([w, h, w, h])).tolist() if ocr_bbox_pixel else []
        xyxy_ratio = (xyxy_pixel / torch.Tensor([w, h, w, h])).tolist()
        ocr_bbox_elem = [{'type': 'text', 'bbox': box, 'interactivity': False, 'content': txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(ocr_bbox_ratio, ocr_text) if int_box_area(box, w, h) > 0]
        xyxy_elem = [{'type': 'icon', 'bbox': box, 'interactivity': True, 'content': None} for box in xyxy_ratio if int_box_area(box, w, h) > 0]
        filtered_boxes_elem = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
        filtered_boxes_sorted = sorted(filtered_boxes_elem, key=lambda x: x['content'] is None)
        starting_idx = next((i for i, box in enumerate(filtered_boxes_sorted) if box['content'] is None), -1)
        filtered_boxes_tensor = torch.tensor([box['bbox'] for box in filtered_boxes_sorted])
        parsed_content_icon = get_parsed_content_icon(filtered_boxes_tensor, starting_idx, image_source_np, caption_model_processor)
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


# --- 3. 创建 Gradio 应用的函数 ---
def create_gradio_app():
    """
    将 gradio_demo.py 的内容封装成一个函数，它返回一个Gradio Blocks应用。
    """
    def process_for_gradio(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz):
        result = run_omniparser_processing_with_image(
            image_input, box_threshold, iou_threshold, use_paddleocr, imgsz
        )
        if result["status"] == "success":
            img_data = base64.b64decode(result["annotated_image_base64"])
            image = Image.open(io.BytesIO(img_data))
            parsed_content_str = '\n'.join([f'Element {i}: ' + str(v) for i, v in enumerate(result["parsed_elements"])])
            return image, parsed_content_str
        else:
            return None, f"Error: {result['message']}"

    with gr.Blocks() as demo:
        gr.Markdown("# OmniParser - Gradio Demo")
        with gr.Row():
            with gr.Column():
                image_input_component = gr.Image(type='pil', label='Upload image')
                box_threshold_component = gr.Slider(label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
                iou_threshold_component = gr.Slider(label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
                use_paddleocr_component = gr.Checkbox(label='Use PaddleOCR', value=True)
                imgsz_component = gr.Slider(label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
                submit_button_component = gr.Button(value='Submit', variant='primary')
            with gr.Column():
                image_output_component = gr.Image(type='pil', label='Image Output')
                text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output', lines=15)

        submit_button_component.click(
            fn=process_for_gradio,
            inputs=[image_input_component, box_threshold_component, iou_threshold_component, use_paddleocr_component, imgsz_component],
            outputs=[image_output_component, text_output_component]
        )
    return demo

# --- 4. 创建并组合 FastAPI 应用 ---
app = FastAPI(
    title="OmniParser API and Demo",
    description="一个同时提供API和Gradio演示的服务。",
    version="2.0.0"
)

# API 端点 (保持不变)
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
    if generate_annotated_image:
        func_to_run = partial(run_omniparser_processing_with_image, image_input=input_image, box_threshold=box_threshold, iou_threshold=iou_threshold, use_paddleocr=use_paddleocr, imgsz=imgsz)
    else:
        func_to_run = partial(run_omniparser_processing_headless, image_input=input_image, box_threshold=box_threshold, iou_threshold=iou_threshold, use_paddleocr=use_paddleocr, imgsz=imgsz)
    
    result = await loop.run_in_executor(None, func_to_run)
    
    if result["status"] == "success":
        return JSONResponse(status_code=200, content=result)
    else:
        return JSONResponse(status_code=500, content=result)

# --- 5. 挂载 Gradio 应用 ---
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# --- 6. 启动服务器 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)