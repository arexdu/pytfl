import numpy as np
import random
import matplotlib.pyplot as plt

##SKlearn LinearRegression Example
random.seed(42)
np.random.seed(42)
ages=[]
for ii in range(100):
    ages.append(random.randint(20,65))
worths=[ii*6.25 + np.random.normal(scale=40.) for ii in ages]

ages=np.reshape(np.array(ages),(len(ages),1))
worths=np.reshape(np.array(worths),(len(worths),1))


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(ages,worths)
print("tom's worth prediction:", reg.predict([[27]]))
print("r-squared score:",reg.score(ages,worths))
print("slope:",reg.coef_)
print("intercept:",reg.intercept_)

plt.scatter(ages,worths)
plt.plot(ages,reg.predict(ages),color='blue',linewidth=3)
plt.xlabel("ages")
plt.ylabel("worths")
plt.show()

