import numpy as np
import matplotlib.pyplot as plt

accuracy1 = np.load('test_sigmoid_1.npy')
accuracy2 = np.load('test_sigmoid_4.npy')
accuracy3 = np.load('test_sigmoid_10.npy')

epoch = accuracy1.size

plt.style.use('_mpl-gallery')
x = np.linspace(1, epoch, epoch)
l1 = plt.plot(x,accuracy1)
l2 = plt.plot(x,accuracy2)
l3 = plt.plot(x,accuracy3)
plt.xlabel("Eporchs")
plt.ylabel("Accuracy")
max_indx1 = np.argmax(accuracy1)
max_indx2 = np.argmax(accuracy2)
max_indx3 = np.argmax(accuracy3)
x_max1 = max_indx1+1
y_max1 = accuracy1[max_indx1]
x_max2 = max_indx2+1
y_max2 = accuracy2[max_indx2]
x_max3 = max_indx3+1
y_max3 = accuracy3[max_indx3]
show_max1='['+str(x_max1)+' '+str(y_max1)+']'
plt.annotate(show_max1,xytext=(x_max1-0.015,y_max1-0.015),xy=(x_max1,y_max1))
show_max2='['+str(x_max2)+' '+str(y_max2)+']'
plt.annotate(show_max2,xytext=(x_max2-0.017,y_max2+0.017),xy=(x_max2,y_max2))
show_max3='['+str(x_max3)+' '+str(y_max3)+']'
plt.annotate(show_max3,xytext=(x_max3-0.009,y_max3+0.009),xy=(x_max3,y_max3))
plt.legend((l1,l2,l3),labels = ['Sigmoid(x), alpha = 1','Sigmoid(x), alpha = 4','Sigmoid(x), alpha = 10'])

plt.show()