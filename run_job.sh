#!/bin/bash
#SBATCH−−job−name=train_waste # Nome do job na fila do SLURM
#SBATCH−−account=f2025hpcvlab00005ubia # Conta associada (CPU)
#SBATCH−−time=00:30:00 # Tempo execucao (20 min.)
#SBATCH−−nodes=1 # Numero de nos ( computadores )
#SBATCH−−ntasks=1 # Numero maximo de tarefas
#SBATCH−−partition=normal−arm # Particao (CPU)
#SBATCH−−cpus−per−task=4 # Numero de CPU cores por tarefa
#SBATCH−−output="output_train_waste . txt " # Nome do ficheiro de saida padrao
#SBATCH−−error="error_train_waste . txt " # Nome do ficheiro de saida erro

# Limpar modulos carregados
module purge
# Ativar conda na arquitetura ARM
source /eb/aarch64/software/Anaconda3/2023.07−2/etc/profile .d/conda.sh
# Ativar o conda env criado
conda activate test−env
# Executar o script python
python /projects/F2025HPCVLAB00005UBI/user/test_waste/train_cnn.py