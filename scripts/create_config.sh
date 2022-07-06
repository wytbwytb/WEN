for split in 2
do
    for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
        for shot in 1 2 3 5 10
        do
           python3 tools/create_config.py --dataset voc --config_root configs/RFS-detection                \
             --shot ${shot} --seed ${seed} --split ${split}
        done
    done
done