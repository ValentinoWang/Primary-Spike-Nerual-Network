syms t
t = linspace(-5,5,1000);
y1 = disrc(t);plot
alpha = 1;
y2 = 1./(1+exp(-alpha*t));
alpha = 5;
y3 = 1./(1+exp(-alpha*t));
alpha = 25;
y4 = 1./(1+exp(-alpha*t));

alpha = 1;
y5 = 1/pi*atan(pi/2*alpha*t)+1/2;
alpha = 5;
y6 = 1/pi*atan(pi/2*alpha*t)+1/2;
alpha = 25;
y7 = 1/pi*atan(pi/2*alpha*t)+1/2;

alpha = 1;
y8 = 1/2*alpha*t./(1+abs(alpha*t))+1/2;
alpha = 5;
y9 = 1/2*alpha*t./(1+abs(alpha*t))+1/2;
alpha = 25;
y10 = 1/2*alpha*t./(1+abs(alpha*t))+1/2;

subplot(1,3,1)
plot(t,y1,t,y2,t,y3,t,y4)
title('Surrogate function: $\frac{1}{1+e^{-\alpha x}}$','Interpreter','latex')
legend('Heaviside','\alpha = 1','\alpha = 5','\alpha = 25','Location','northwest')

subplot(1,3,2)
plot(t,y1,t,y5,t,y6,t,y7)
title('Surrogate function: $\frac{1}{\pi} \arctan \left(\frac{\pi}{2} \alpha x\right)+\frac{1}{2}$','Interpreter','latex')
legend('Heaviside','\alpha = 1','\alpha = 5','\alpha = 25','Location','northwest')

subplot(1,3,3)
plot(t,y1,t,y8,t,y9,t,y10)
title('Surrogate function: $g(x)=\frac{\alpha}{2(1+|\alpha x|)^{2}}$','Interpreter','latex')
legend('Heaviside','\alpha = 1','\alpha = 5','\alpha = 25','Location','northwest')