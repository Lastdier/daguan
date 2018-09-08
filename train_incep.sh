python DNNmain_fold.py main --max_epoch=30 --plot_every=100 --env='inception' --weight=1 --model='CNNText_inception'  --batch-size=150 --inception_dim=512 --lr=0.001 --lr2=0.0005 --lr_decay=0.8 --decay_every=300  --weight-decay=0 --type_='word' --debug-file='/tmp/debug'  --linear-hidden-size=2000 --vocab_size = 498681 --save_model_path = 'CNNText_inception.pt' --embedding_path='vector.300dim' --embedding_dim=300 --content_seq_len=1780