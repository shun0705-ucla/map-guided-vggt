# demo.py \
#    --image input/1636351539931534.png input/1636351649229239.png input/1636351945927477.png \
#    --config configs/mg3-base.yaml \
#    --checkpoint checkpoints/mg3/mg3_base_init.pth \
#    --outdir output \
#    --resolution 448 \

python demo.py \
    --image input/1636351539931534.png input/1636351945927477.png \
    --config configs/evggt_6ch.yaml \
    --checkpoint checkpoints/evggt.pt \
    --outdir output \
    --resolution 518 \
    --save_checkpoint checkpoints/evggt_6ch.pt