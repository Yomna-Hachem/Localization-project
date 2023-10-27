import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt 

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        # Initialize kalman variables
        
        
        # Define sampling time
        self.dt = 0.1
       
        # Intial State
        self.state = np.matrix([[0], [0], [0], [0]])
        
        # initial covariance matrix
        self.P = np.identity(4)
        
        # Define the State Transition Matrix F
        self.F = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control/ Input transition Matrix G
        self.G = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        
        self.u = self.dt* self.state[2]
        
        #Initial Process Noise Covariance
        self.Q = np.identity(4)
        
        
        #initialize the plot array
        
        self.estimated_x =[]
        self.estimated_y=[]
        self.real_x=[]
        self.real_y=[]
        
        
        
        
        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,'/odom_noise', self.odom_callback, 1)
        
        #publish the estimated reading
        self.estimated_pub=self.create_publisher(Odometry, "/odom_estimated",1)

    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        
        noisy_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
        
        # Prediction step
        
        
        self.state = np.dot(self.F, self.state) + np.dot(self.G, self.u)
        
    
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    

        
        # Update step
        
        self.measurement_noise_covariance = np.identity(4) * 0.1  
    

        kalman_gain = np.dot(np.dot(self.P, self.H.T),
        inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.measurement_noise_covariance))

        self.state = self.state + np.dot(kalman_gain, (noisy_position - np.dot(self.H, self.state)))
        self.state_covariance = np.dot((np.identity(2) - np.dot(kalman_gain, self.H)), self.P)


        # Create an Odometry message with the estimated state
        estimated_odom = Odometry()
        estimated_odom.header = msg.header
        estimated_odom.pose.pose.position.x = self.state[0]
        estimated_odom.pose.pose.position.y = self.state[1]

        # Publish the estimated odometry
        self.estimated_pub.publish(estimated_odom)
        
        self.estimated_x.append(self.state[0])
        self.estimated_y.append(self.state[1])
        
    


    plt.plot(estimated_x, estimated_y)
    plt.show()


pass    

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
