set -eu
R=/root/autodl-tmp/ouro-e6a
cd "$R"
echo '54847eae9e8b501d53c5d310c3117ad3b43964aa2f58fbfd5c2711c21ed159be  E6B/fas5090/E6B_FAS_TRANSFER_FREEZE.json' | sha256sum -c -
test -f E6A/FAS_DEVELOPMENT_FROZEN.json
test ! -f full_answer/test/COMPLETE.json
venv/bin/python -u E6B/E6_SHARED_FULL_ANSWER_CAPTURE_v2.py --model /root/autodl-tmp/ouro-e4-e5/model --inputs E6A/FULL_ANSWER_test.jsonl --out full_answer/test > fas_test_capture.log 2>&1
for item in 'fas_char5090 scores' 'fas_sum5090 raw_sum' 'fas_token5090 mean_token'; do
 set -- $item
 venv/bin/python -u E6A/evaluate_capture_v2.py --capture full_answer --labels /root/autodl-tmp/ouro-e4-e5/inputs/labels.json --out "E6A/$1" --aggregation "$2" --phase test
done
venv/bin/python -u E6A/finalize_e6a.py
venv/bin/python -u E6A/plot_e6a.py
echo FULL_ANSWER_TEST_AND_A_REPORTS_COMPLETE
