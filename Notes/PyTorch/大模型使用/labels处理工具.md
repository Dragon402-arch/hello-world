#### labels 处理

- 方式一

  ```python
  import torch
  
  input_ids = torch.tensor([[2, 2, 2, 2, 2, 2, 2, 64790, 64792, 53456,
                             33071, 51026, 30910, 31732, 35795, 2]])
  labels = input_ids.clone()
  
  pad_token_id = 2
  labels[labels == pad_token_id] = -100
  print(labels)
  ```

  

- 方式二

  ```python
  import torch
  
  input_ids = torch.tensor([[2, 2, 2, 2, 2, 2, 2, 64790, 64792, 53456,
                             33071, 51026, 30910, 31732, 35795, 2]])
  labels = input_ids.clone()
  
  pad_token_id = 2
  labels = torch.where(labels != pad_token_id,labels,-100)
  print(labels)
  ```

  

### LoRA权重保存与加载

#### Transformers方式保存与加载

- 保存

  ```python
  model.save_pretrained("./lora_model")
  ```

- 加载

  ```python
  pretrained_model_path = "/home/lis/algProjects/pretrained_models/chatglm2-6b/"
  lora_model_path = "/home/lis/algProjects/finetuneChatGLM/trainChatGLM2/lora_model/"
  
  model = AutoModel.from_pretrained(pretrained_model_path, trust_remote_code=True, device_map="auto").half()
  model = PeftModel.from_pretrained(model, lora_model_path).half()
  
  ```

#### Torch方式保存与加载

- 保存

  ```python
  saved_params = {
      k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
  }
  torch.save(saved_params, args.checkpoint_filepath)
  ```

  

- 加载

  ```python
  def get_model(pretrained_model_path):
      peft_config = LoraConfig(
          task_type=TaskType.CAUSAL_LM,
          inference_mode=True,
          r=8,
          lora_alpha=32,
          lora_dropout=0.1,
          target_modules=['query_key_value', 'dense_h_to_4h', 'dense_4h_to_h'],
          fan_in_fan_out=False
      )
      model = AutoModel.from_pretrained(pretrained_model_path, trust_remote_code=True, device_map="auto").half()
      model = get_peft_model(model, peft_config)
      return model
  
  model = get_model(pretrained_model_path)
  peft_filepath = "/home/lis/algProjects/finetuneChatGLM/trainChatGLM2/chatglm2_lora.pth"
  model.load_state_dict(torch.load(peft_filepath), strict=False)
  
  ```