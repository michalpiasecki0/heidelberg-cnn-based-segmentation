conda activate ml-base
python train.py -m dataset=dic_augmented,fluo_augmented,phc_augmented,dic,fluo,phc \
                   model=unet,unet_wider,unet_shallow \
                   model.loss_weights.dice=0,1 \
                   model.loss_weights.cross_entropy=0,1 \
