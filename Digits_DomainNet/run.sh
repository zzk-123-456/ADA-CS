
# DomainNet
for ADA_AL in 'CLUE' 'entropy' 'margin' 'coreset' 'leastConfidence' 'BADGE' 'MHP' ; do
  for total_budget_second in  600 800 1000 1200 1400; do
    total_budget_cal=$((5000 - $total_budget_second))
    python train.py --load_from_cfg True  --cfg_file config/domainnet/clue_mme.yml  --al_strat $ADA_AL --da_strat 'mme' \
    --total_budget_second $total_budget_second --total_budget $total_budget_cal --source 'clipart' --target 'painting'
    done
done

for total_budget_second in  600 800 1000 1200 1400; do
  total_budget_cal=$((5000 - $total_budget_second))
  python train.py --load_from_cfg True  --cfg_file config/domainnet/clue_mme.yml  --al_strat AADA --da_strat 'dann' \
  --total_budget_second $total_budget_second --total_budget $total_budget_cal --source 'clipart' --target 'painting'
  done


# Digits
python train.py --load_from_cfg True  --cfg_file config/digits/clue_mme.yml  --al_strat 'CLUE' --da_strat 'mme' \
--total_budget_second 50 --total_budget 250 --num_rounds 3 --num_rounds_2 1 --source 'svhn' --target 'mnist'