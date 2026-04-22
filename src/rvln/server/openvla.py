"""
OpenVLA inference server.

Loads the OpenVLA model, serves action predictions via Flask HTTP.
Vendored from UAV-Flow/OpenVLA-UAV (commit 0114801).
"""

import base64
import logging
import os
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
        self.device_mode = cfg.get("device", "cuda")

        if self.device_mode not in ("cuda", "cpu", "auto"):
            raise ValueError(
                f"Unknown device mode: {self.device_mode!r} (expected 'cuda', 'cpu', or 'auto')"
            )
        if self.device_mode in ("cuda", "auto") and not torch.cuda.is_available():
            raise RuntimeError(
                f"device={self.device_mode!r} requires CUDA, but torch.cuda.is_available() is False. "
                "Use --device cpu to run without a GPU."
            )

        self.model_path = cfg.get("model_path")
        log.info(f"Loading model: {self.model_path} (device={self.device_mode})")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        log.info(f"Processor type: {type(self.processor)}")

        load_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.device_mode == "cuda":
            load_kwargs["attn_implementation"] = "flash_attention_2"
            load_kwargs["torch_dtype"] = torch.bfloat16
            self.compute_dtype = torch.bfloat16
            self.device = torch.device(f"cuda:{self.gpu_id}")
        elif self.device_mode == "cpu":
            load_kwargs["torch_dtype"] = torch.float32
            self.compute_dtype = torch.float32
            self.device = torch.device("cpu")
        else:  # auto: split across GPU + CPU RAM via accelerate
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
            self.compute_dtype = torch.bfloat16
            self.device = None  # set after load from first parameter

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        if self.device_mode != "auto":
            self.model = self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device
        log.info(f"VLA model type: {type(self.model)}")

        self.model.eval()

        self.unnorm_key = cfg.get("unnorm_key", "sim")
        self.do_sample = cfg.get("do_sample", False)
        if torch.cuda.is_available():
            self._alltime_peak_vram = torch.cuda.max_memory_allocated(self.gpu_id) / (1024 ** 3)
        else:
            self._alltime_peak_vram = 0.0

        if self.device_mode == "auto":
            self._validate_device_split()
            if torch.cuda.is_available():
                self._alltime_peak_vram = max(
                    self._alltime_peak_vram,
                    torch.cuda.max_memory_allocated(self.gpu_id) / (1024 ** 3),
                )

        self.app = Flask(__name__)
        self._setup_routes()
        self.port = cfg.get("http_port", 5000)

    def _validate_device_split(self):
        """Run a dummy forward pass to verify the GPU/CPU split survives runtime memory usage.

        ``device_map="auto"`` only accounts for static weight sizes. Activations,
        KV cache, and intermediate tensors created during ``generate()`` also
        consume GPU memory. This method runs a realistic inference call right
        after loading so an OOM surfaces at startup rather than mid-flight.

        If the dummy pass OOMs, the model is reloaded with ``max_memory`` reduced
        by 20% to leave more headroom. If it still fails after one retry, the
        error propagates so the caller knows the model cannot fit.
        """
        log.info("Validating device split with dummy inference pass...")
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        dummy_prompt = "In: Current State: 0.0,0.0,0.0,0.0, What action should the uav take to hover?\nOut:"

        try:
            with torch.inference_mode():
                self.act(dummy_image, dummy_prompt)
            log.info("Device split validation passed.")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning(
                "OOM during dummy inference. Reloading model with reduced GPU memory budget."
            )
            gpu_id = self.gpu_id
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            reduced = int(total * 0.8)
            log.info(
                f"Reducing GPU {gpu_id} max_memory from "
                f"{total / 1e9:.1f} GB to {reduced / 1e9:.1f} GB"
            )

            del self.model
            torch.cuda.empty_cache()

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={gpu_id: reduced, "cpu": "64GiB"},
            )
            self.model.eval()
            self.device = next(self.model.parameters()).device

            with torch.inference_mode():
                self.act(dummy_image, dummy_prompt)
            log.info("Device split validation passed after memory reduction.")

    def _log_memory_usage(self, request_peak_vram_gb=None):
        """Log memory usage after an inference call."""
        rss_bytes = 0
        peak_rss_bytes = 0
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_bytes = int(line.split()[1]) * 1024
                    elif line.startswith("VmHWM:"):
                        peak_rss_bytes = int(line.split()[1]) * 1024
        except OSError:
            pass
        rss_gb = rss_bytes / (1024 ** 3)
        peak_rss_gb = peak_rss_bytes / (1024 ** 3)

        if torch.cuda.is_available():
            gpu_id = self.gpu_id
            current_vram = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
            reserved_vram = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
            if request_peak_vram_gb is not None:
                self._alltime_peak_vram = max(self._alltime_peak_vram, request_peak_vram_gb)
            req_str = f"request_peak={request_peak_vram_gb:.2f} GB, " if request_peak_vram_gb is not None else ""
            log.info(
                f"Memory: VRAM {req_str}alltime_peak={self._alltime_peak_vram:.2f} GB, "
                f"current={current_vram:.2f} GB, reserved={reserved_vram:.2f} GB | "
                f"CPU RSS={rss_gb:.2f} GB, peak_RSS={peak_rss_gb:.2f} GB"
            )
        else:
            log.info(f"Memory: CPU RSS={rss_gb:.2f} GB, peak_RSS={peak_rss_gb:.2f} GB")

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

                request_peak = None
                if torch.cuda.is_available():
                    request_peak = torch.cuda.max_memory_allocated(self.gpu_id) / (1024 ** 3)
                self._log_memory_usage(request_peak_vram_gb=request_peak)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.gpu_id)

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

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                vram_total = torch.cuda.get_device_properties(self.gpu_id).total_mem / (1024 ** 3)
                log.error(
                    "CUDA out of memory during inference (%.1f GB total VRAM). "
                    "Try restarting the server with --device auto (offloads layers "
                    "to CPU RAM) or --device cpu.",
                    vram_total,
                )
                return jsonify({
                    "status": "error",
                    "error_type": "cuda_oom",
                    "message": (
                        f"CUDA out of memory ({vram_total:.1f} GB total VRAM). "
                        "Restart the server with --device auto or --device cpu."
                    ),
                }), 507

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({
                    "status": "error",
                    "error_type": "unknown",
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
        inputs = inputs.to(self.device, dtype=self.compute_dtype)

        pred_action = self.model.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=self.do_sample,
        )
        return pred_action
