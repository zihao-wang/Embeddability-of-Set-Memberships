# iterate for m = 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
# iterate for k = 2,3,4,5

for m in 16 32 64 128 256 512 1024 2048 4096 8192
do
    echo "m = $m" for tabular embedding and inner product
    python3 med_solver.py -M $m -K 2 -E tabular -F inner_product --cuda 0 &
    python3 med_solver.py -M $m -K 3 -E tabular -F inner_product --cuda 1 &
    python3 med_solver.py -M $m -K 4 -E tabular -F inner_product --cuda 2 &
    python3 med_solver.py -M $m -K 5 -E tabular -F inner_product --cuda 3 &
    wait
done