#!/usr/bin/env python

import sys
import rospy
import pypozyx
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu

# publish positioning and IMU data from Pozyx tag
# this is only set up for one (local) tag doing 3D positioning

# for gyroscope, acceleration, and magnetometer, tag must be calibrated every time the tag is plugged back in

class uwbOdom:
    def __init__(self, base_frame = "uwb", robot_name = "robot"):
        self.name = rospy.get_name()
        self.base_frame = base_frame
        self.robot_ns = robot_name
        self.tag_ready = False
        self.anchors_added = False
        self.anchors_ready = False

        filt_name = rospy.get_param(self.name + '/position_filter')
        if(filt_name == 'none'):
            self.filter_type = pypozyx.PozyxConstants.FILTER_TYPE_NONE
        elif(filt_name == 'low-pass'):
            self.filter_type = pypozyx.PozyxConstants.FILTER_TYPE_FIR
        elif(filt_name == 'average'):
            self.filter_type = pypozyx.PozyxConstants.FILTER_TYPE_MOVING_AVERAGE
        elif(filt_name == 'median'):
            self.filter_type = pypozyx.PozyxConstants.FILTER_TYPE_MOVING_MEDIAN
        else:
            rospy.warn('Invalid filter type selected. Defaulting to none')
            self.filter_type = pypozyx.PozyxConstants.FILTER_TYPE_NONE
        self.filter_strength = min(max(rospy.get_param(self.name + '/position_filter_strength'), 0), 15)
        self.pozyx_algo = pypozyx.PozyxConstants.POSITIONING_ALGORITHM_TRACKING if rospy.get_param(self.name + "/use_pozyx_tracking") else pypozyx.PozyxConstants.POSITIONING_ALGORITHM_UWB_ONLY
        self.range_protocol = pypozyx.PozyxConstants.RANGE_PROTOCOL_PRECISION if (rospy.get_param(self.name + "/range_protocol") == 'precision') else pypozyx.PozyxConstants.RANGE_PROTOCOL_FAST

        self.port = pypozyx.get_first_pozyx_serial_port()
        self.anchors = [pypozyx.DeviceCoordinates(anchor['id'], 1,
                                pypozyx.Coordinates(anchor['loc'][0], anchor['loc'][1], anchor['loc'][2]))
                                for anchor in rospy.get_param(self.name + "/anchors")]

        self.tag = None
        self.position = pypozyx.Coordinates()
        self.quat = pypozyx.Quaternion()

        if(rospy.get_param(self.name + '/use_pozyx_tracking')):
            self.pos_orient_cov = np.array([[ 6.40902840e-05, -8.85211920e-05, -4.62694880e-05,-1.47375241e-31, -8.35896737e-30, -2.83074847e-28],
                                            [-8.85211920e-05,  3.62656496e-04,  2.37780344e-04, 4.11625155e-32,  2.33483106e-30,  7.93720288e-29],
                                            [-4.62694880e-05,  2.37780344e-04,  2.16460416e-04, 1.57174372e-31,  8.90939506e-30,  3.02020131e-28],
                                            [-1.47375241e-31,  4.11625155e-32,  1.57174372e-31, 4.33334237e-34,  2.45556068e-32,  8.32001736e-31],
                                            [-8.35896737e-30,  2.33483106e-30,  8.90939506e-30, 2.45556068e-32,  1.39148438e-30,  4.71467650e-29],
                                            [-2.83074847e-28,  7.93720288e-29,  3.02020131e-28, 8.32001736e-31,  4.71467650e-29,  1.59744333e-27]])
        
            self.ang_vel_cov = np.array([[ 3.06564580e-04, -3.60517451e-07,  6.62400035e-07],
                                         [-3.60517451e-07,  1.54048534e-04, -9.18114853e-07],
                                         [ 6.62400035e-07, -9.18114853e-07,  2.28925934e-04]])

            self.accel_cov = np.array([[ 4.06176808e-04, -2.06157240e-05, -2.06881937e-05],
                                       [-2.06157240e-05,  1.60101437e-04, -3.34098952e-06],
                                       [-2.06881937e-05, -3.34098952e-06,  2.39190662e-04]])

        else:
            self.pos_orient_cov = np.array([[ 9.77488560e-03, -2.25701945e-02,  3.11398920e-02, 8.18443189e-32,  3.41971202e-31, -1.45150407e-31],
                                            [-2.25701945e-02,  2.41238051e-01, -9.12996809e-02, 1.08073944e-31,  4.49650716e-31, -2.02737253e-31],
                                            [ 3.11398920e-02, -9.12996809e-02,  1.79017093e-01, 1.45347622e-31,  6.04267453e-31, -2.70973721e-31],
                                            [ 8.18443189e-32,  1.08073944e-31,  1.45347622e-31, 6.77084746e-32,  2.81667254e-31, -1.26389153e-31],
                                            [ 3.41971202e-31,  4.49650716e-31,  6.04267453e-31, 2.81667254e-31,  1.17173578e-30, -5.25778875e-31],
                                            [-1.45150407e-31, -2.02737253e-31, -2.70973721e-31, -1.26389153e-31, -5.25778875e-31,  2.35926418e-31]])

            self.ang_vel_cov = np.array([[ 1.53979549e-04, -1.21735426e-06,  1.27300748e-06],
                                         [-1.21735426e-06,  2.86414821e-06,  1.01729911e-06],
                                         [ 1.27300748e-06,  1.01729911e-06,  7.73233975e-05]])
            
            self.accel_cov = np.array([[ 4.32463046e-04, -3.78754530e-05,  7.87761721e-06],
                                       [-3.78754530e-05,  1.30642988e-04,  2.24079506e-06],
                                       [ 7.87761721e-06,  2.24079506e-06,  2.88763802e-04]])

        #self.cov = [pypozyx.SingleRegister(size = 2), pypozyx.SingleRegister(size = 2), pypozyx.SingleRegister(size = 2), 
        #            pypozyx.SingleRegister(size = 2), pypozyx.SingleRegister(size = 2), pypozyx.SingleRegister(size = 2)]
        #self.cov_registers = [pypozyx.PozyxRegisters.POSITIONING_ERROR_X, pypozyx.PozyxRegisters.POSITIONING_ERROR_Y, pypozyx.PozyxRegisters.POSITIONING_ERROR_Z,
        #                      pypozyx.PozyxRegisters.POSITIONING_ERROR_XY, pypozyx.PozyxRegisters.POSITIONING_ERROR_XZ, pypozyx.PozyxRegisters.POSITIONING_ERROR_YZ]

        self.ang_vel = pypozyx.AngularVelocity()
        self.accel = pypozyx.Acceleration()

        self.posePub = rospy.Publisher(self.robot_ns + '/uwb/odom', PoseWithCovarianceStamped, queue_size=10)
        self.imuPub = rospy.Publisher(self.robot_ns + '/uwb/imu', Imu, queue_size=10)

    def connectTag(self):
        # open connection
        if(self.port is None):
            rospy.loginfo_throttle(10, 'No Pozyx device detected.')
            self.port = pypozyx.get_first_pozyx_serial_port()
        else:
            rospy.loginfo('Opening serial connection at port ' + self.port)
            try:
                self.tag = pypozyx.PozyxSerial(self.port)
                rospy.loginfo('Serial connection opened.')
            except:
                rospy.loginfo('Failed to open serial connection')

            # tag runs self test automatically at startup
            # see register datasheet to diagnose specific test failures
            test = pypozyx.SingleRegister()
            self.tag.regRead(pypozyx.PozyxRegisters.SELFTEST_RESULT, test)
            if(test == 0x3f):
                rospy.loginfo('Pozyx tag self test success.')
            else:
                rospy.logerr('Pozyx tag self test failed. Recommend checking components individually.')

            # calibrate motion sensors
            self.calibrateTag()

            # set tracking algorithm
            algo = pypozyx.SingleRegister()
            self.tag.getPositionAlgorithm(algo)
            if(algo != self.pozyx_algo):
                self.tag.setPositionAlgorithm(self.pozyx_algo, pypozyx.PozyxConstants.DIMENSION_3D)
            #rospy.loginfo('Using ' + ('Pozyx tracking' if self.pozyx_algo == pypozyx.PozyxConstants.POSITIONING_ALGORITHM_TRACKING else 'UWB only'))

            # set additional filtering
            filter_dat = pypozyx.FilterData()
            self.tag.getPositionFilterData(filter_dat)
            if((self.filter_type != filter_dat.filter_type) | (self.filter_strength != filter_dat.filter_strength)):
                self.tag.setPositionFilter(self.filter_type, self.filter_strength)

            # set ranging protocol
            rp = pypozyx.SingleRegister()
            self.tag.getRangingProtocol(rp)
            if(self.range_protocol != rp):
                self.tag.setRangingProtocol(self.range_protocol)
            

            self.tag_ready = True

    def connectAnchors(self):
        rospy.loginfo('Setting up anchors...')
        success = True
        self.tag.clearDevices() # clears previous devices from tag list

        for anchor in self.anchors:
            success &= self.tag.addDevice(anchor)

        success &= self.tag.setSelectionOfAnchors(pypozyx.POZYX_ANCHOR_SEL_AUTO, len(self.anchors))

        # check
        list_size = pypozyx.SingleRegister()
        self.tag.getDeviceListSize(list_size)
        list_cont = pypozyx.DeviceList(list_size = list_size[0])
        self.tag.getDeviceIds(list_cont)

        if(success):
            rospy.loginfo('Added anchors with IDs: ' + ', '.join(hex(i) for i in list_cont.data))
            self.anchors_added = True

    def checkAnchorStatus(self):
        status_all = 0

        good_status_msg = ''
        bad_status_msg = ''

        for anchor in self.anchors:
            device_range = pypozyx.DeviceRange()
            status = self.tag.doRanging(anchor.network_id, device_range)
            if(status == pypozyx.POZYX_SUCCESS):
                status_all += 1
                good_status_msg += 'Anchor ' + hex(anchor.network_id) + ' is ' + str(device_range.distance) + 'mm away.\n'
            else:
                bad_status_msg += 'Anchor ' + hex(anchor.network_id) + ' is turned off or out of range.\n'

        if(len(good_status_msg)):
            rospy.loginfo_throttle(5, good_status_msg)
        if(len(bad_status_msg)):
            rospy.logwarn_throttle(5, bad_status_msg)

        if(status_all < 4):
            rospy.logwarn_throttle(5, 'Not enough anchors in range for 3D positioning (' + str(status_all) + '/4).')
            self.anchors_ready = False
        else:
            rospy.loginfo_throttle(5, str(status_all) + ' anchors are in range.')
            self.anchors_ready = True

    def calibrateTag(self):
        calib_status = pypozyx.SingleRegister()
        self.tag.regRead(pypozyx.PozyxRegisters.CALIBRATION_STATUS, calib_status)

        # gyroscope
        while((calib_status.value & 0b00110000) >> 4 != 3): 
            rospy.loginfo_once('Calibrating gyroscope. Hold Pozyx tag in stable position.')
            self.tag.regRead(pypozyx.PozyxRegisters.CALIBRATION_STATUS, calib_status)
        rospy.loginfo('Gyroscope calibration complete.')

        # accelerometer
        while((calib_status.value & 0b00001100) >> 2 != 3): 
            rospy.loginfo_once('Calibrating accelerometer. Hold Pozyx tag in stable position for a few seconds, then move to a new position, changing which axis is up.')
            self.tag.regRead(pypozyx.PozyxRegisters.CALIBRATION_STATUS, calib_status)
        rospy.loginfo('Accelerometer calibration complete.')

        # magnetometer
        while((calib_status.value & 0b00000011) != 3): 
            rospy.loginfo_once('Calibrating magnetometer. Make a figure-eight with the Pozyx tag.')
            self.tag.regRead(pypozyx.PozyxRegisters.CALIBRATION_STATUS, calib_status)
        rospy.loginfo('Magnetometer calibration complete.')
    
    def publishUWB(self):
        # PoseWithCovarianceStamped
        #       Header
        #       pose [position (x, y, z), orientation (x, y, z, w)]
        #       covariance (row-major 6x6 covariance as [x, y, z, r, p, y])
        # uses x, y, z; err_x, err_y, err_z; err xy, err xz, err yz; quat_x, quat_y, quat_z, quat_w (no cov known)
        # converts from mm to m; mm^2 to m^2

        pos_success = self.tag.doPositioning(self.position)
        rot_success = self.tag.getNormalizedQuaternion(self.quat)

        # although there's a set of registers for covariance, it's not actually reported
        # cov_success = 1
        # for register, dat in zip(self.cov_registers, self.cov):
        #     cov_success &= self.tag.regRead(register, dat)
        # cov_matrix = np.array([[self.cov[0][0]/(1000**2), self.cov[3][0]/(1000**2), self.cov[4][0]/(1000**2), 0, 0, 0],
        #                        [self.cov[3][0]/(1000**2), self.cov[1][0]/(1000**2), self.cov[5][0]/(1000**2), 0, 0, 0],
        #                        [self.cov[4][0]/(1000**2), self.cov[5][0]/(1000**2), self.cov[2][0]/(1000**2), 0, 0, 0],
        #                        [0, 0, 0, 0, 0, 0],
        #                        [0, 0, 0, 0, 0, 0],
        #                        [0, 0, 0, 0, 0, 0]])

        cov_matrix = np.diag([.1, .1, .1, .1, .1, .1])

        if((pos_success & rot_success)):
            msgPos = PoseWithCovarianceStamped()
            msgPos.header.stamp = rospy.Time.now()
            msgPos.header.frame_id = self.base_frame

            msgPos.pose.pose.position.x = self.position.x/1000
            msgPos.pose.pose.position.y = self.position.y/1000
            msgPos.pose.pose.position.z = self.position.z/1000
            msgPos.pose.pose.orientation.x = self.quat.x
            msgPos.pose.pose.orientation.y = self.quat.y
            msgPos.pose.pose.orientation.z = self.quat.z
            msgPos.pose.pose.orientation.w = self.quat.w
            msgPos.pose.covariance = cov_matrix.flatten('C') #float64[36] row-major representation of 6x6 cov matrix

            self.posePub.publish(msgPos)

        # IMU
        #       Header
        #       orientation [x, y, z, w]
        #       orientation covariance (row-major 3x3 covariance as (r, p, y))
        #       angular_velocity (r_d, p_d, y_z)
        #       angular_velocity_covariance (row-major 3x3 covariance as (r, p, y))
        #       linear_acceleration (x_dd, y_dd, z_dd)
        #       linear_acceleration_covariance (row-major 3x3 covariance as (r, p, y))
        # uses gyro_x, gyro_y, gyro_z (no cov known); lia_x, lia_y, lia_z (no cov known)
        # because orientation is already included in the pose message, we set covariances to -1 
        # converts from degrees/sec to rad/sec; milli-g to m/s^2

        ang_success = self.tag.getAngularVelocity_dps(self.ang_vel)
        accel_success = self.tag.getAcceleration_mg(self.accel)

        if((ang_success & accel_success)):
            msgIMU = Imu()
            msgIMU.header.stamp = rospy.Time.now()
            msgIMU.header.frame_id = self.base_frame

            msgIMU.orientation.x = 0 #self.quat.x
            msgIMU.orientation.y = 0 #self.quat.y
            msgIMU.orientation.z = 0 #self.quat.z
            msgIMU.orientation.w = 0 #self.quat.w
            msgIMU.orientation_covariance = np.diag([-1, 0, 0]).flatten('C')

            msgIMU.angular_velocity.x = self.ang_vel.x*np.pi/180
            msgIMU.angular_velocity.y = self.ang_vel.y*np.pi/180
            msgIMU.angular_velocity.z = self.ang_vel.z*np.pi/180
            msgIMU.angular_velocity_covariance = np.diag([0.1, 0.1, 0.1]).flatten('C')

            msgIMU.linear_acceleration.x = self.accel.x*(9.81/1000)
            msgIMU.linear_acceleration.y = self.accel.y*(9.81/1000)
            msgIMU.linear_acceleration.z = self.accel.z*(9.81/1000)
            msgIMU.linear_acceleration_covariance = np.diag([0.1, 0.1, 0.1]).flatten('C')

            self.imuPub.publish(msgIMU)

    def calibrate_variance(self): # rough estimate for variance of each, based on holding tag still
        rospy.loginfo('Checking variances. Keep tag still.')
        wait = input('Press Enter when ready to start.')

        position_m = []
        orientation_rpy = []
        ang_vel_rad_s = []
        accel_m_s2 = []

        ii = 1000

        while(((len(position_m) < ii) | (len(orientation_rpy) < ii)) | ((len(ang_vel_rad_s) < ii) | (len(accel_m_s2) < ii))):
            pos_success = self.tag.doPositioning(self.position)
            rot_success = self.tag.getNormalizedQuaternion(self.quat)
            if((pos_success) & (rot_success)):
                position_m.append([self.position.x/1000, self.position.y/1000, self.position.z/1000])
                orientation_rpy.append(np.asarray(Rotation.from_quat([self.quat.x, self.quat.y, self.quat.z, self.quat.w]).as_euler('xyz')))
            ang_success = self.tag.getAngularVelocity_dps(self.ang_vel)
            if(ang_success):
                ang_vel_rad_s.append([self.ang_vel.x*np.pi/180, self.ang_vel.y*np.pi/180, self.ang_vel.z*np.pi/180])
            accel_success = self.tag.getAcceleration_mg(self.accel)
            if(accel_success):
                accel_m_s2.append([self.accel.x*(9.81/1000), self.accel.y*(9.81/1000), self.accel.z*(9.81/1000)])

        rospy.loginfo('Done collecting.')

        pos_orient = np.hstack((np.array(position_m), np.array(orientation_rpy)))
        position_cov = np.cov(pos_orient.T, bias=True)
        ang_vel_rad_s = np.array(ang_vel_rad_s)
        ang_vel_cov = np.cov(ang_vel_rad_s.T, bias=True)
        accel_m_s2 = np.array(accel_m_s2)
        accel_cov = np.cov(accel_m_s2.T, bias=True)

        #rospy.loginfo(pos_orient)
        rospy.loginfo('Position & RPY: \n' + repr(position_cov))
        rospy.loginfo('Angular Velocity: \n' + repr(ang_vel_cov))
        rospy.loginfo('Acceleration: \n' + repr(accel_cov))

if __name__ == '__main__':
    rospy.init_node('uwb_odom')
    myargv = rospy.myargv(argv = sys.argv)
    node = uwbOdom(myargv[1], myargv[2])
    rate = rospy.Rate(50)            

    while not rospy.is_shutdown():
        if(node.anchors_ready):
            if(not rospy.get_param(rospy.get_name() + '/cov_check')):
                node.publishUWB()
            else:
                node.calibrate_variance()
                break
        elif(not node.tag_ready):
            node.connectTag()
        elif(not node.anchors_added):
            node.connectAnchors()
        else:
            node.checkAnchorStatus()

        rate.sleep()