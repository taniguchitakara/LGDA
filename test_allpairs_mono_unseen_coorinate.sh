./tools/dist_test.sh \
/large/ttani_2/bhrl/configs/manga/BHRL_allpairs_mono.py \
/large/ttani_2/bhrl/work_dirs/manga/allpairs/trained_tensor_gaussian/ablation/coordinate/epoch_30.pth \
1 \
--out ./allpairs/ablation/coordinate.pkl \
--eval bbox \
--test_seen_classes