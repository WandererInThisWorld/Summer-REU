function u = idct2 (uhat)
u = idct ( idct (uhat ).').';
