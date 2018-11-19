CC=nvcc
CFLAGS=-O3 -lm -arch sm_20
SOURCES=cuda_Cell_Detailed.cu
all: basic

basic:
	$(CC) $(SOURCES) $(CFLAGS) $(OPTFLAG) -o cuda_Cell_Detailed
l:
	$(CC) $(SOURCES) $(CFLAGS) -Dlqt -o cuda_Cell_Detailed
t:
	$(CC) $(SOURCES) $(CFLAGS) -Dtapsi -o cuda_Cell_Detailed
tl: lt
lt:
	$(CC) $(SOURCES) $(CFLAGS) -Dtapsi -Dlqt -o cuda_Cell_Detailed
i:
	$(CC) $(SOURCES) $(CFLAGS) -Diso2 -o cuda_Cell_Detailed
il: li
li:
	$(CC) $(SOURCES) $(CFLAGS) -Diso2 -Dlqt -o cuda_Cell_Detailed
it: ti
ti:
	$(CC) $(SOURCES) $(CFLAGS) -Diso2 -Dtapsi -o cuda_Cell_Detailed
lit: lti
til: lti
tli: lti
itl: lti
ilt: lti
lti:	
	$(CC) $(SOURCES) $(CFLAGS) -Diso2 -Dtapsi -Dlqt -o cuda_Cell_Detailed