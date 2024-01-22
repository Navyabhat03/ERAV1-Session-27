# ERA-SESSION27

🤗[**Space Link**]



### Tasks:
1. :heavy_check_mark: Use OpenAssistant dataset.
2. :heavy_check_mark: Finetune Microsoft Phi2 model.
3. :heavy_check_mark: Use QLoRA stratergy.
4. :heavy_check_mark: Create an App on HF space using finetuned model.

## Phi2 Model Description:
```python
PhiForCausalLM(
  (transformer): PhiModel(
    (embd): Embedding(
      (wte): Embedding(51200, 2560)
      (drop): Dropout(p=0.0, inplace=False)
    )
    (h): ModuleList(
      (0-31): 32 x ParallelBlock(
        (ln): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.1, inplace=False)
        (mixer): MHA(
          (rotary_emb): RotaryEmbedding()
          (Wqkv): Linear4bit(in_features=2560, out_features=7680, bias=True)
          (out_proj): Linear4bit(in_features=2560, out_features=2560, bias=True)
          (inner_attn): SelfAttention(
            (drop): Dropout(p=0.0, inplace=False)
          )
          (inner_cross_attn): CrossAttention(
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (mlp): MLP(
          (fc1): Linear4bit(in_features=2560, out_features=10240, bias=True)
          (fc2): Linear4bit(in_features=10240, out_features=2560, bias=True)
          (act): NewGELUActivation()
        )
      )
    )
  )
  (lm_head): CausalLMHead(
    (ln): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (linear): Linear(in_features=2560, out_features=51200, bias=True)
  )
  (loss): CausalLMLoss(
    (loss_fct): CrossEntropyLoss()
  )
)
```

### Training Output
```python
TrainOutput(global_step=500, training_loss=1.711106170654297, metrics={'train_runtime': 5222.3118, 'train_samples_per_second': 1.532, 'train_steps_per_second': 0.096, 'total_flos': 3.293667738832896e+16, 'train_loss': 1.711106170654297, 'epoch': 0.81})
```
### Loss vs Steps Logs
![image](https://github.com/Navyabhat03/ERAV1-Session-27/assets/60884505/02b22bc0-593f-4513-9a85-b3adc497bda7)

## Sample Results:
![image](https://github.com/Navyabhat03/ERAV1-Session-27/assets/60884505/f116509e-a578-4bb4-8cc6-43a68852b6f1)

![image](https://github.com/Navyabhat03/ERAV1-Session-27/assets/60884505/79852734-235b-48af-ae52-d305a261e279)

## Gradio UI:
![image](https://github.com/Navyabhat03/ERAV1-Session-27/assets/60884505/6e49f571-fca7-459c-8423-cd3b1dd62d9f)

