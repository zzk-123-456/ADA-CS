## Usage 

### Digits and DomainNet

To run the method on the Digits and DomainNet datasets use the code from Digits_DomainNet.If you wanted to use the ADA-CS method on the Digits dataset, you could run the following code:

```
python train.py --load_from_cfg True  --cfg_file config/digits/clue_mme.yml  --al_strat 'CLUE' --da_strat 'mme' \
--total_budget_second 50 --total_budget 250 --num_rounds 3 --num_rounds_2 1 --source 'svhn' --target 'mnist'
```
