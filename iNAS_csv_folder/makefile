all:
	gcc -std=c99 -I../include/ -o iNAS_csv iNAS_csv.c iNAS_csv_test_suite.c -L../include/ -l:libkeras2c.a -lm
	gcc -std=c99 -I../include/ -o r10_iNAS_csv iNAS_csv.c r10_iNAS_csv_test.c -L../include/ -l:libkeras2c.a -lm

clean:
	rm iNAS_csv r10_iNAS_csv