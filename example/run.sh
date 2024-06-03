gcc -mavx2 -DENABLE_AVX2 -I ../ ./WFA_vect_example.c ../build/libwfa.a -o ./WAF_vect_example.out
./WAF_vect_example.out -p ./data_sample.seq -l 1000