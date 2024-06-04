using Anomaly, Plots, Random

sol,pl = driver();
savefig(pl,"/Users/steve/Desktop/anomaly.pdf")

# for publication figure
rstate = Xoshiro(0xa242084f5b199bea, 0x53a1df5ce28618fd, 0x89250c8d5127cb4e, 0x6b5f53c46218257b, 0xc5ed5ca8a7cca17e);
sol,pl = driver(;rstate=rstate);
savefig(pl,"/Users/steve/Desktop/anomaly2.pdf")