"""
OpenVLA inference server.

Loads the OpenVLA model, serves action predictions via Flask HTTP.
Vendored from UAV-Flow/OpenVLA-UAV (commit 0114801).
"""

import base64
import logging
import time
from io import BytesIO

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

log = logging.getLogger(__name__)


class OpenVLAActionAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gpu_id = cfg.get("gpu_id", 0)
        self.device = torch.device(f"cuda:{self.gpu_id}")

        self.model_path = cfg.get("model_path")
        log.info(f"Loading model: {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        log.info(f"Processor type: {type(self.processor)}")

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        log.info(f"VLA model type: {type(self.model)}")

        self.model.eval()

        self.app = Flask(__name__)
        self._setup_routes()
        self.port = cfg.get("http_port", 5000)

        self.unnorm_key = cfg.get("unnorm_key", "sim")
        self.do_sample = cfg.get("do_sample", False)

    def _setup_routes(self):
        @self.app.route("/predict", methods=["POST"])
        def predict():
            try:
                data = request.json

                img_bytes = base64.b64decode(data["image"])
                image = Image.open(BytesIO(img_bytes))

                instruction = data["instr"]
                proprio = np.array(data["proprio"], dtype=np.float32)
                proprio_str = ",".join([str(round(x, 1)) for x in proprio])

                prompt = f"In: Current State: {proprio_str}, What action should the uav take to {instruction}?\nOut:"
                log.info(f"Prompt: {prompt}")

                start_time = time.time()
                with torch.inference_mode():
                    pred_action = self.act(image, prompt)
                log.info(f"Inference time: {time.time() - start_time:.3f}s")

                pred_action = pred_action[None, :]
                pred_action_ori = pred_action.copy()
                log.info(f"Raw predicted action: {pred_action_ori}")

                current_yaw = np.deg2rad(proprio[-1])
                current_pos = proprio[0:3]
                cos_yaw = np.cos(current_yaw)
                sin_yaw = np.sin(current_yaw)
                R = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1],
                ])

                pred_action[0, 0:3] = R @ pred_action[0, 0:3]
                pred_action[0, 0:3] = current_pos + pred_action[0, 0:3]
                pred_action[0, -1] = pred_action[0, -1] + current_yaw

                return jsonify({
                    "status": "success",
                    "action": pred_action.tolist(),
                    "action_ori": pred_action_ori.tolist(),
                    "message": "Action generated successfully",
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({
                    "status": "error",
                    "message": str(e) + traceback.format_exc(),
                }), 500

    def run(self):
        """Start the HTTP server."""
        log.info(f"Starting HTTP server on port {self.port}")
        self.app.run(host="0.0.0.0", port=self.port)

    def act(self, image, prompt):
        """Run action inference.

        Args:
            image: PIL.Image instance
            prompt: prompt text
        Returns:
            pred_action: [4] numpy array
        """
        inputs = self.processor(prompt, image)
        inputs = inputs.to(self.device, dtype=torch.bfloat16)

        pred_action = self.model.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=self.do_sample,
        )
        return pred_action
