
for model_name in Reformer
do
    root_path_name=./dataset/
    data_path_name=modified_dataset.csv
    model_id_name=custom2
    data_name=custom2

    pred_len=192
    for seq_len in 48 96 144 192 288 384 480 576 720
    do
        python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 16 \
        --enc_in 23 \
        --d_model 512 \
        --dropout 0.5 \
        --train_epochs 50 \
        --patience 10 \
        --dec_in 23 \
        --c_out 3 \
        --itr 1 --batch_size 8 --learning_rate 0.001
       
    done
done


# TCN-MLP PatchMixer Linear TIDE TSMixer Informer iTransformer LSTM GRU SegRNN TCN
# RPTM PatchMixer Linear TIDE TSMixer Informer iTransformer LSTM GRU