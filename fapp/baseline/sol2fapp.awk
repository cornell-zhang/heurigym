
# Translate toulbar2 solution in CFN format (-s=3 or -w=3) to ROADEF2001 output format for fappeval checker

# gcc -o fappeval fappeval.c
# python3 fapp.py exemple1.in 3 | awk -f ./sol2fapp.awk - exemple1

BEGIN {
 problem = ARGV[2];
 ARGC -= 1;
}

{print $0}

/(X[0-9]+=[(][0-9]+, [-]?1[)])+/ {
 print "RP",0,1,0,0,0,1,0,0,0,1,0,0,0 > problem ".out";
 gsub("="," ",$0); gsub("X","",$0); gsub("[(]","",$0); gsub("[)]","",$0); gsub(",","",$0);
 for (i=1; i<=NF; i+=3) print "AL",$i,$(i+1),$(i+2) >> problem ".out";
 close(problem ".out");
 system("./fappeval " problem);
}

