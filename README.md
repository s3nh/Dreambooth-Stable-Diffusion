# 1 st version was to complex

accelerate is useless, I am not targeting into 8bit adam stuff. 
text_encoder is a must have, otherwise, results are not regular. 
triton does not want to build on 3.9, so xformer are not used. 

# Example runs 

accelerate launch train_dreambooth.py \
 --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks s3nhchihiro style" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=9000
  
  
  


accelerate launch train_dreambooth.py --pretrained_model_name_or_path=$MODEL_NAME  --train_text_encoder --instance_data_dir=$INSTANCE_DIR --class_data_dir=$CLASS_DIR --output_dir=$OUTPUT_DIR --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="Illustation of a person, spirited away style" --class_prompt="Illustration fo a person"  --resolution=512 --train_batch_size=1 --gradient_checkpointing --learning_rate=2e-6  --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=200 --max_train_steps=1500   
 
 
 !git clone https://github.com/openai/triton.git
 cd triton/python
 pip install -e .
 
 
 
 # beks
 
 
 accelerate launch train_dreambooth.py --pretrained_model_name_or_path=$MODEL_NAME  --train_text_encoder --instance_data_dir=$INSTANCE_DIR --class_data_dir=$CLASS_DIR --output_dir=$OUTPUT_DIR --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="Portrait of a person" --class_prompt="Portrait of a person, beksinski style"  --resolution=512 --train_batch_size=1 --gradient_checkpointing --learning_rate=2e-6  --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=200 --max_train_steps=9000
 
 
 
export MODEL_NAME="runwayml/stable-diffusion-v1-5" #@param {type:"string"}
export OUTPUT_DIR="/notebooks/output_dir"
export CLASS_DIR="/notebooks/class_prompt/reg/person"
export INSTANCE_PROMPT="Illustation of person, beks style"
export CLASS_PROMPT="Illustration of a person"
export INSTANCE_DIR="/notebo"
 
 
 accelerate launch train_dreambooth.py --pretrained_model_name_or_path=$MODEL_NAME  --train_text_encoder --instance_data_dir=$INSTANCE_DIR --class_data_dir=$CLASS_DIR --output_dir=$OUTPUT_DIR --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="Portrait of a person, arcane style" --class_prompt="Portrait of a person"  --resolution=512 --train_batch_size=1 --gradient_checkpointing --learning_rate=2e-6  --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=200 --max_train_steps=200
 
 
 
