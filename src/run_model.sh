now=$(date +"%Y%m%d_%H%M%S")
logdir=./train_log/exp_$now
mkdir -p "$logdir"
echo "output dir: $logdir"

cd ..
export PYTHONPATH=$PYTHONPATH:$PWD
cd src || exit
export PYTHONPATH=$PYTHONPATH:$PWD
echo "PYTHONPATH: $PYTHONPATH"

# Download pretrained weights
if [ ! -d "pretrained" ]
then
      mkdir pretrained
      cd pretrained || exit
      wget "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
      wget "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
      wget "https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar"
      wget "https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_m-56M-224-84.0.pth.tar"
      gdown "https://drive.google.com/uc?id=1m1QZRfo7P3_PqR3cTC760VT2xCqn8Ggm"
      gdown "https://drive.google.com/uc?id=1NyqP8uctLkuGJzItW9CeYQW6jj23rXxT"
      cd ..
fi


while [[ $# -gt 0 ]]
do
        key=$1
        case $key in
                --IMNET)
                        IMNET=YES
                        EXPERIMENT_NAME=$2
                        NUM_GPUS=$3
                        shift
                        shift
                        shift
                        ;;
                --CIFAR)
                        CIFAR=YES
                        MODELCONFIG=$2
                        shift
                        shift
                        ;;
                *)
                        echo "Unknown option {$1}"
                        shift
                        ;;
        esac
done

if [[ -n "$IMNET" ]]; then
        CONFIG_PATH="../experiments-results-analysis/experiments/${EXPERIMENT_NAME}_config.yaml"
        runcommand="python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
            --use_env main.py \
            --output_dir ${logdir} \
            --config_path ${CONFIG_PATH}"
        echo "running: $runcommand"
        eval $runcommand
fi
if [[ -n "$CIFAR" ]]; then
        datapath=$HOME/CIFAR100

        # Run command
        case $MODELCONFIG in
                # ViT
                vit_base_patch16_224)
                        batchsize=64
                        inputsize=224
                        ;;
                vit_tiny_patch16_224)
                        batchsize=128
                        inputsize=224
                        ;;
                *)
                        echo "Unknown option {$1}"
                        ;;
        esac
        runcommand="python -m torch.distributed.launch --nproc_per_node=4 \
            --use_env main.py \
            --model $MODELCONFIG \
            --data_set CIFAR  \
            --batch_size $batchsize \
            --input_size $inputsize \
            --data_path $datapath \
            --output_dir $logdir \
            --epochs 30 \
            --base_rate 0.7 \
            --lr 1e-3 \
            --experiment_name imnet_$MODELCONFIG"
        echo "running: $runcommand"
        eval $runcommand
fi

echo "output dir for the last exp: $logdir"
