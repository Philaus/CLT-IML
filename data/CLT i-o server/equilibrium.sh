#!/bin/bash

# Place the modified inequ and eq_transp.F files in the workspace folder.
# The inequ parameters have already been initialized.
rm *.o
make -f mkeq_tr
cp eqb1 eqxz
./goeq_tr


sed -i '5c\03        0.70      .00000001 300.0     1800      0.0                 .200' inequ
sed -i '6c\04        +0.00     1.0000   +0.2093    .00000001 2.0       01.60     .400' inequ
sed -i '10c\08        0.0       0.0        0.0       128.      97.      32.0       +1.0' inequ
sed -i '12c\13        0.0       +0.000    0.0      + 2.00      05.0      01.0       3.0' inequ
cp eqb1 eqxz
./goeq_tr


sed -i '5c\03        0.70      .00000001 300.0     2000      0.0                 .200' inequ
sed -i '6c\04        +0.00     1.0000   +0.4093    .00000001 2.0       01.60     .400' inequ
cp eqb1 eqxz
./goeq_tr

sed -i '6c\04        +0.00     1.0000   +0.5093    .00000001 2.0       01.60     .400' inequ
cp eqb1 eqxz
./goeq_tr

ctrans -d ps.mono ploteq >ploteq.ps
ps2pdf ploteq.ps
