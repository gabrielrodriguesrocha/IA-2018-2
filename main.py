import sys
from kmeans import kmeans
from slink import slink
from avglink import avglink

if (len(sys.argv) == 5):
	if (sys.argv[1] == '-A'):
		avglink(sys.argv[2], sys.argv[3], sys.argv[4])
	elif (sys.argv[1] == '-S'):
		slink(sys.argv[2], sys.argv[3], sys.argv[4])
	elif (sys.argv[1] == '-K'):
		kmeans(sys.argv[2], sys.argv[3], sys.argv[4])
	else:
		print ('USAGE: main -[A|S|K] FILE [MIN_K|MAX_ITER] [MAX_K|NUM_CLUSTERS]')
		print ('Where: -A runs AVGLINK, -S runs SLINK, -K runs KMEANS')
		print ('MIN_K and MAX_K are arguments for AVGLINK and SLINK')
		print ('MAX_ITER and NUM_CLUSTERS are arguments for KMEANS')
else:
	print ('USAGE: main -[A|S|K] FILE [MIN_K|MAX_ITER] [MAX_K|NUM_CLUSTERS]')
	print ('Where: -A runs AVGLINK, -S runs SLINK, -K runs KMEANS')
	print ('MIN_K and MAX_K are arguments for AVGLINK and SLINK')
	print ('MAX_ITER and NUM_CLUSTERS are arguments for KMEANS')