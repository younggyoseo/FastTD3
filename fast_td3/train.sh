for seed in {2..5}
do
    python fast_td3/train.py \
        --env_name G1JoystickFlatTerrain \
        --exp_name FastTD3 \
        --render_interval 5000 \
        --seed $seed \
        --agent fasttd3
done

# 这个脚本不能用simbav2，必定会报错