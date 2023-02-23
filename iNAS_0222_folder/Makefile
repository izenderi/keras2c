all:
	gcc -std=c99 -I../include/ -o iNAS_0222 iNAS_0222.c iNAS_0222_test_suite.c -L../include/ -l:libkeras2c.a -lm
	gcc -std=c99 -I../include/ -o iNAS_clean iNAS_clean.c iNAS_clean_test_suite.c -L../include/ -l:libkeras2c.a -lm
clean:
	rm -f iNAS_0222