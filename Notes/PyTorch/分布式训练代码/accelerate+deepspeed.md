### DeepSpeed支持的功能

1. **Optimizer state** partitioning (ZeRO stage 1)

2. **Gradient** partitioning (ZeRO stage 2)：主要用于训练部分，推理部分无关

3. **Parameter** partitioning (ZeRO stage 3)：可以用于训练，也可以用于推理，因此其允许使用多个GPU加载单个GPU无法加载的模型。

4. Custom mixed precision training handling

5. A range of fast CUDA-extension-based **optimizers**

6. ZeRO-Offload to CPU and Disk/NVMe

   [示例代码](https://github.com/huggingface/accelerate/tree/main/examples/by_feature)

   [使用指南](https://blog.csdn.net/weixin_43301333/article/details/127237122)

### Accelerate 集成 DeepSpeed方式

- Integration of the DeepSpeed features via `deepspeed config file` specification in `accelerate config` .

  该方式需要自定义配置文件或使用我们的模板，可以支持DeepSpeed的所有核心功能。更为灵活和功能强大。

#### 1、通过 deepspeed_plugin 方式集成

该方式支持DeepSpeed功能的子集，并为其余配置使用默认选项。

- **ZeRO Stage-2 DeepSpeed Plugin Example**

  - `acce_config.yaml `文件内容

    ```yaml
    compute_environment: LOCAL_MACHINE
    deepspeed_config:
     gradient_accumulation_steps: 1
     gradient_clipping: 1.0
     offload_optimizer_device: none
     offload_param_device: none
     zero3_init_flag: true
     zero_stage: 2
    distributed_type: DEEPSPEED
    fsdp_config: {}
    machine_rank: 0
    main_process_ip: null
    main_process_port: null
    main_training_function: main
    mixed_precision: fp16
    num_machines: 1
    num_processes: 2
    use_cpu: false
    ```

  - 终端命令

    ```shell
    accelerate launch  --config_file ./acce_config.yaml nlp_example.py
    ```

- **ZeRO Stage-3 with CPU Offload DeepSpeed Plugin Example**

  - 配置文件内容

    ```yaml
    compute_environment: LOCAL_MACHINE
    deepspeed_config:
      gradient_accumulation_steps: 1
      gradient_clipping: 1.0
      offload_optimizer_device: cpu
      offload_param_device: cpu
      zero3_init_flag: true
      zero3_save_16bit_model: true
      zero_stage: 3
    distributed_type: DEEPSPEED
    fsdp_config: {}
    machine_rank: 0
    main_process_ip: null
    main_process_port: null
    main_training_function: main
    mixed_precision: fp16
    num_machines: 1
    num_processes: 2
    use_cpu: false
    ```

  - 终端命令

    ```shell
    accelerate launch  --config_file ./acce_config.yaml nlp_example.py
    ```

#### 2、通过 deepspeed config 文件方式集成

- **ZeRO Stage-2 DeepSpeed Config File Example**

  ```yaml
  compute_environment: LOCAL_MACHINE
  deepspeed_config:
   deepspeed_config_file: /home/ubuntu/accelerate/examples/configs/deepspeed_config_templates/zero_stage2_config.json
   zero3_init_flag: true
  distributed_type: DEEPSPEED
  fsdp_config: {}
  machine_rank: 0
  main_process_ip: null
  main_process_port: null
  main_training_function: main
  mixed_precision: fp16
  num_machines: 1
  num_processes: 2
  use_cpu: false
  ```

  

- **ZeRO Stage-3 DeepSpeed Config File Example**

  ```yaml
  compute_environment: LOCAL_MACHINE
  deepspeed_config:
   deepspeed_config_file: /home/ubuntu/accelerate/examples/configs/deepspeed_config_templates/zero_stage3_offload_config.json
   zero3_init_flag: true
  distributed_type: DEEPSPEED
  fsdp_config: {}
  machine_rank: 0
  main_process_ip: null
  main_process_port: null
  main_training_function: main
  mixed_precision: fp16
  num_machines: 1
  num_processes: 2
  use_cpu: false
  ```

- 保存和加载模型权重：

  -  ZeRO Stage-1 and Stage-2：保存和加载模型权重与原来没有变化

  - ZeRO Stage-3：保存时需要去除包装

    - 直接保存 16bit model weights 方便以后直接加载使用

      - DeepSpeed Config File：zero_optimization.stage3_gather_16bit_weights_on_model_save: true
      -  DeepSpeed Plugin：zero3_save_16bit_model=True

    - 代码示例

      1.  16bit model weights

         ```python
         # 保存 16bit model weights
         unwrapped_model = accelerator.unwrap_model(model)
         
         unwrapped_model.save_pretrained(
             args.output_dir,
             is_main_process=accelerator.is_main_process,
             save_function=accelerator.save,
             state_dict=accelerator.get_state_dict(model),
         )
         ```

      2.  32bit model weights

         ```python
         # 保存 32bit model weights
         success = model.save_checkpoint(PATH, ckpt_id, checkpoint_state_dict)
         status_msg = "checkpointing: PATH={}, ckpt_id={}".format(PATH, ckpt_id)
         if success:
             logging.info(f"Success {status_msg}")
         else:
             logging.warning(f"Failure {status_msg}")
         ```

         - 离线获取最终模型权重

           ```shell
           $ cd /path/to/checkpoint_dir
           
           # pytorch_model.bin文件为指定模型保存文件的名称，可以自行定义，可以改为trained_model.bin
           $ ./zero_to_fp32.py . pytorch_model.bin
           # Processing zero checkpoint at global_step1
           # Detected checkpoint of type zero stage 3, world_size: 2
           # Saving fp32 state dict to pytorch_model.bin (total_numel=60506624)
           ```

         - 直接加载未转换前的权重

           ```python
           from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
           
           unwrapped_model = accelerator.unwrap_model(model)
           fp32_model = load_state_dict_from_zero_checkpoint(unwrapped_model, checkpoint_dir)
           ```

           

      

      

### Accelerate 常用命令

- accelerate env
- accelerate test
- accelerate config
- accelerate launch 
  - accelerate launch  -h :查看可传参数
- 原则是，**能直接多卡训练，就不要用ZeRO；能用ZeRO-2就不要用ZeRO-3.**