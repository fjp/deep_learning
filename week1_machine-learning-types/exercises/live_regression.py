
# coding: utf-8

# In[25]:

from numpy import *


# In[49]:

def compute_error_for_line_given_points(b, m, points):
    #initialize error at 0
    totalError = 0
    for i in range(0, len(points)):
        #get the x value
        x = points[i, 0]
        #get the y value
        y = points[i, 1]
        #get the difference, square it, add it to the total
        totalError += (y - (m* x + b))**2
    
    #get the average
    return totalError/float(len(points))


# In[56]:

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #starting b and m
    b = starting_b
    m = starting_m
    
    #gradient descent
    for i in range(num_iterations):
        #update b and m with the new more accurate b an m by performing
        #this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


# In[73]:

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    
    N = float(len(points))
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #direciton with respect to b and m
        #computing partial derivatives of our error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
        
    #update our b and m values using our partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    
    return [new_b, new_m]


# In[76]:

def run():
    
    #Step 1 - collect our data
    points = genfromtxt('data.csv', delimiter=',')
    
    #Step 2 - define our hyperparameters
    #how fast should our model converge?
    learning_rate = 0.0001
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    
    #Step 3 - train our model
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

    


# In[77]:

if __name__ == '__main__':
    run()

