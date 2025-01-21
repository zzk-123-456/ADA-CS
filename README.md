## Usage 

### Digits and DomainNet

To run the method on the Digits and DomainNet datasets use the code from ```Digits_DomainNet```.If you wanted to use the ADA-CS method on the Digits dataset, you could run the following code:

```
python train.py --load_from_cfg True  --cfg_file config/digits/clue_mme.yml  --al_strat 'CLUE' --da_strat 'mme' \
--total_budget_second 50 --total_budget 250 --num_rounds 3 --num_rounds_2 1 --source 'svhn' --target 'mnist'
```
In this code, we use the ADA method CLUE to complete the first stage of sample selection. A total of three rounds of sample selection are carried out, and a total of 100 samples are selected in each round. In the first round, 50 samples are selected by ADA-CS.


If you wanted to use the ADA-CS method on the DomainNet dataset, you could run the following code:
```
  python train.py --load_from_cfg True  --cfg_file config/domainnet/clue_mme.yml  --al_strat 'CLUE' --da_strat 'mme' \
  --total_budget_second 1000 --total_budget 4000 --source 'clipart' --target 'painting'
```

More running commands can be found in ```run.sh```

To run the method on the Office-Home RSUT datasets use the code from ```Digits_DomainNet```.If you wanted to use the ADA-CS method on the Office-Home RSUT dataset, you could run the following code:

```
python main.py --cfg configs/officehome_RSUT.yaml --gpu 0 --log log/oh_RSUT/LADA \
ADA.AL 'AADA' ADA.DA dann LADA.S_M 5 ADA.BUDGET 0.1 \
SECOND.IF True SECOND.BUDGET_2 0.05
```

More running commands can be found in ```run_baseline.sh```
