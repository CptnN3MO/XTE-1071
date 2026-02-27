data none
abund wilm
xsect vern
mo tbabs*(nsatmos+pegpwrlw)

newpar 1 3.18344 -1
newpar 2 6
newpar 3 1.6 -1
newpar 4 16 -1
newpar 5 8.8 -1
newpar 6 1 -1
newpar 7 1 -1
newpar 8 .5 -1
newpar 9 10 -1
newpar 10 .2

data 1:1 26976_grp.pi 2:2 26977_grp.pi 3:3 26978_grp.pi 4:4 obs4_grp.pi 5:5 26980_grp.pi

setpl e
ig bad
fit
untie 12 22 32 42
untie 10 20 30 40 50
fit
freeze 10 20 30 40 50 
renorm
fit
uncer 1.0 2 12 22 32 42
show free
flux 0.5 10 err 100 90
show frozen
