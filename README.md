# 初版命令行方式

本地后面启动，先用这个最小命令。

  在 PowerShell 里跑，推荐先切到：

  cd G:\AR_Diffusion\Bard-VL\sglang-diffusion\python

  然后启动：

  python -m sglang.launch_server `
    --model-path G:\你的BARD模型目录 `
    --tokenizer-path G:\你的BARD模型目录 `
    --dllm-algorithm Dflash `
    --dllm-verifier-model-path G:\你的Qwen3_vl模型目录 `
    --dllm-verifier-threshold -5.0 `
    --host 0.0.0.0 `
    --port 30000 `
    --tp-size 1

  需要替换的只有两处：

  - G:\BARD模型目录
  - G:\Qwen3_vl模型目录

  如果是多卡，比如 4 卡，就把最后改成：

  --tp-size 4

  如果你还想带一个算法配置文件，再加这一项：

  --dllm-algorithm-config G:\你的配置文件\dflash.yaml

  一个可用的 dflash.yaml 最小例子可以先写成：

  threshold: 0.95
  denoising_steps: 4
  verifier_threshold: -5.0
  pad_target_penalty: 1.0

  所以完整版本会是：

  python -m sglang.launch_server `
    --model-path G:\你的BARD模型目录 `
    --tokenizer-path G:\你的BARD模型目录 `
    --dllm-algorithm Dflash `
    --dllm-algorithm-config G:\你的配置文件\dflash.yaml `
    --dllm-verifier-model-path G:\你的Qwen3_vl模型目录 `
    --dllm-verifier-threshold -5.0 `
    --host 0.0.0.0 `
    --port 30000 `
    --tp-size 1

  几个注意点：

  - 不要再加 --speculative-algorithm，这条新链路和 speculative 是分开的。
  - 第一版默认按 greedy 跑，先别开复杂采样参数。
  - --enable-multimodal 不用专门写，默认就是开的。
  - 如果你的模型目录不是标准 HF 结构，再单独看报错补参数。

  如果你愿意，我下一条可以直接按你本地的真实模型目录名，帮你把这条命令改成一条可以直接复制运行的最终版本。
