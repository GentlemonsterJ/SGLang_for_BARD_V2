

# Dstar-VL

Download model weights:
```
huggingface-cli download cbyzju/Dstar-VL-B4-Mask-4B-Instruct --local-dir pretrained_models/Dstar/Dstar-VL-B4-Mask-4B-Instruct

huggingface-cli download cbyzju/Dstar-VL-B8-Mask-4B-Instruct --local-dir pretrained_models/Dstar/Dstar-VL-B8-Mask-4B-Instruct

huggingface-cli download cbyzju/Dstar-VL-B16-Mask-4B-Instruct --local-dir pretrained_models/Dstar/Dstar-VL-B16-Mask-4B-Instruct

huggingface-cli download cbyzju/Dstar-VL-B32-Mask-4B-Instruct --local-dir pretrained_models/Dstar/Dstar-VL-B32-Mask-4B-Instruct

huggingface-cli download cbyzju/Dstar-VL-B4-Mask-8B-Instruct --local-dir pretrained_models/Dstar/Dstar-VL-B4-Mask-8B-Instruct
```

Start a server:
```
nohup env SGLANG_USE_CUDA_IPC_TRANSPORT=1 python3 -m sglang.launch_server \
    --model-path pretrained_models/Dstar/Dstar-VL-B4-Mask-4B-Instruct \
    --served-model-name Dstar-VL \
    --enable-multimodal \
    --dllm-algorithm low_confidence_dynamic \
    --dllm-algorithm-config configs/dstar_low_confidence_dynamic.yaml \
    --keep-mm-feature-on-device \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    >/tmp/dstar_vl.log 2>&1 &
```

Send request:
```
curl -s http://127.0.0.1:30000/v1/chat/completions     -H "Content-Type: application/json"     --data-binary '{
      "model": "Dstar-VL",
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Describe this image."},
            {
              "type": "image_url",
              "image_url": {
                "url": "file:///inspire/hdd/project/chineseculture/public/chenbaoyou/workspace/sglang/assets/puzzle.jpg"
              }
            }
          ]
        }
      ],
      "max_tokens": 4096, "return_step_map":true,"return_step_confidence_map":true,"return_token_logprobs":true
    }'
```