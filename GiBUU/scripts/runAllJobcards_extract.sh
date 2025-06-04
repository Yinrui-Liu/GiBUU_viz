#!/bin/bash
for i in `ls *rep`;
do
        j=`basename -s ".rep" "$i"`
        o=`grep -c "BUU simulation: finished" $i`
        t=`awk '{ if (/avgtext/) { gsub("...elapsed",""); print $3; } }' $i`
        m=`du -BM $j| awk '{ print $1 }'`

	[ -z "$t" ] && t="-running-"

	printf "%1s %9s %6s # %s\n" $o $t $m $j
done
