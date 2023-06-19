function F = LotkaVolterrafun2(t,V,a,b,c,d)
X = V(1); Y= V(2);
F(1) = a*X - b*X.*Y;
F(2) = c*X.*Y - d*Y;
F = F';