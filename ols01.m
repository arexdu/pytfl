x=[1896,1900,1904,1906,1908,1912,1920,1924,1928,1932,1936,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008]
t = [12, 11, 11, 11.2, 10.8, 10.8, 10.8, 10.6, 10.8, 10.3, 10.3, 10.3, 10.4, 10.5, 10.2, 10, 9.95, 10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85, 9.69]
n = 27
xa = sum(x)/n
ta = sum(t)/n
xta = sum(x.*t)/n
x2a = sum(x.*x)/n
w1 = (xta - xa*ta)/(x2a - xa * xa)
w0 = ta - w1 * xa
plot(x,t,'ro')
hold on
xx=1896:2008
plot(xx,w0+w1*xx)