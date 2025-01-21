#for ADA_DA in  'mme' 'ft'; do
#  for ADA_AL in 'entropy' 'margin' 'coreset' 'leastConfidence' 'BADGE' 'AADA' 'CLUE' 'MHP'; do
#    python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA ADA.AL $ADA_AL ADA.DA $ADA_DA LADA.S_M 5 ADA.BUDGET 0.05 \
#     SECOND.IF False
#  done
#done

# without tem and weight

#for ADA_AL in 'LAS' 'CLUE' 'entropy' 'margin' 'coreset' 'leastConfidence' 'BADGE' 'MHP' ; do
#  python main.py --cfg configs/officehome_RSUT2.yaml --gpu 0 --log log/oh_RSUT/LADA \
#  ADA.AL $ADA_AL ADA.DA mme LADA.S_M 5 ADA.BUDGET 0.1 \
#  SECOND.IF True SECOND.BUDGET_2 0.05
#done
#
#python main.py --cfg configs/officehome_RSUT2.yaml --gpu 0 --log log/oh_RSUT/LADA \
#ADA.AL 'AADA' ADA.DA dann LADA.S_M 5 ADA.BUDGET 0.1 \
#SECOND.IF True SECOND.BUDGET_2 0.05


python main.py --cfg configs/officehome_RSUT2.yaml --gpu 0 --log log/oh_RSUT/LADA \
ADA.AL LAS ADA.DA mme LADA.S_M 5 ADA.BUDGET 0.1 \
SECOND.IF False SECOND.BUDGET_2 0.05

python main.py --cfg configs/officehome_RSUT2.yaml --gpu 0 --log log/oh_RSUT/LADA \
ADA.AL CLUE ADA.DA mme LADA.S_M 5 ADA.BUDGET 0.1 \
SECOND.IF False SECOND.BUDGET_2 0.05