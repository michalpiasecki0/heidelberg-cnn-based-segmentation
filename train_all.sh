conda activate ml-base
python train.py -m dataset=dic_augmented,fluo_augmented,phc_augmented \
                   model=unet,unet_wider \
                   model.loss_weights.dice=0,1 \
                   model.loss_weights.cross_entropy=0,1 \
