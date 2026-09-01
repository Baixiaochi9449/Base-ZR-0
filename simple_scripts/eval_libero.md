此次评估使用两个核心脚本：

  - 模型服务端：server.py:18
  - LIBERO 评估客户端：evaluation/libero_eval/run_libero_eval.py:42

  完整运行配置记录在 result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202/run_manifest.yaml。

  先连接服务器：

  ssh -p 25408 lq@10.82.1.223
  cd /opt/data/private/lq/ZR-0

  RUN_DIR=/opt/data/private/lq/ZR-0/result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202
  MODEL_DIR=/opt/data/private/lq/models/ZR-0-libero
  SERVER_PY=/opt/data/private/lq/.conda/envs/zr0-eval/bin/python
  CLIENT_PY=/opt/data/private/lq/.conda/envs/zr0-libero-eval/bin/python

  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1

  启动四个模型服务端：

  CUDA_VISIBLE_DEVICES=0 "$SERVER_PY" -u server.py \
    --dataset_entry demo_data.libero_v21 \
    --ckpt_dir "$MODEL_DIR" \
    --inference_mode direct_action \
    --port 8100 \
    > "$RUN_DIR/logs/server_libero_spatial.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=1 "$SERVER_PY" -u server.py \
    --dataset_entry demo_data.libero_v21 \
    --ckpt_dir "$MODEL_DIR" \
    --inference_mode direct_action \
    --port 8001 \
    > "$RUN_DIR/logs/server_libero_object.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=2 "$SERVER_PY" -u server.py \
    --dataset_entry demo_data.libero_v21 \
    --ckpt_dir "$MODEL_DIR" \
    --inference_mode direct_action \
    --port 8002 \
    > "$RUN_DIR/logs/server_libero_goal.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=3 "$SERVER_PY" -u server.py \
    --dataset_entry demo_data.libero_v21 \
    --ckpt_dir "$MODEL_DIR" \
    --inference_mode direct_action \
    --port 8103 \
    > "$RUN_DIR/logs/server_libero_10.log" 2>&1 &

  配置无头 LIBERO 客户端：

  export LIBERO_CONFIG_PATH="$RUN_DIR/env/libero-config"
  export MUJOCO_GL=osmesa
  export PYOPENGL_PLATFORM=osmesa
  export LD_LIBRARY_PATH=/opt/data/private/lq/.conda/envs/zr0-libero-eval/usr/lib/x86_64-linux-gnu:/opt/data/private/lq/.conda/envs/zr0-libero-eval/lib

  启动四组评估：

  "$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
    --args.task-suite-name libero_spatial \
    --args.port 8100 \
    --args.video-out-path "$RUN_DIR/videos/libero_spatial" \
    > "$RUN_DIR/logs/client_libero_spatial.attempt1.log" 2>&1 &

  "$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
    --args.task-suite-name libero_object \
    --args.port 8001 \
    --args.video-out-path "$RUN_DIR/videos/libero_object" \
    > "$RUN_DIR/logs/client_libero_object.attempt1.log" 2>&1 &

  "$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
    --args.task-suite-name libero_goal \
    --args.port 8002 \
    --args.video-out-path "$RUN_DIR/videos/libero_goal" \
    > "$RUN_DIR/logs/client_libero_goal.attempt1.log" 2>&1 &

  "$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
    --args.task-suite-name libero_10 \
    --args.port 8103 \
    --args.video-out-path "$RUN_DIR/videos/libero_10" \
    > "$RUN_DIR/logs/client_libero_10.attempt1.log" 2>&1 &

  没有额外覆盖官方评估参数，脚本默认使用：图像 448×448、render 256×256、action chunk/replan 10、去噪 5 步、每任务 50 次、客户端 seed 7、服务端 seed 42。各 suite 最大步数分别为 280/280/300/520。

  其中 8100 和 8103 是实际完成评估使用的端口，因为原计划的 8000 和 8003 当时已被占用。重新运行时应换一个新的 RUN_DIR，避免覆盖已有结果。