#!/usr/bin/env python

import sys
import rospy
import pypozyx
import numpy as np

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu

# publish positioning and IMU data from Pozyx tag
# this is only set up for one (local) tag doing 3D positioning

# for gyroscope, acceleration, and magnetometer, tag must be calibrated every time the tag is plugged back in

class uwbOdom:
    def __init__(self):
        self.name = rospy.get_name()
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
        self.cov = [pypozyx.SingleRegister(size = 2), pypozyx.SingleRegister(size = 2), pypozyx.SingleRegister(size = 2), 
                    pypozyx.SingleRegister(size = 2), pypozyx.SingleRegister(size = 2), pypozyx.SingleRegister(size = 2)]
        self.cov_registers = [pypozyx.PozyxRegisters.POSITIONING_ERROR_X, pypozyx.PozyxRegisters.POSITIONING_ERROR_Y, pypozyx.PozyxRegisters.POSITIONING_ERROR_Z,
                              pypozyx.PozyxRegisters.POSITIONING_ERROR_XY, pypozyx.PozyxRegisters.POSITIONING_ERROR_XZ, pypozyx.PozyxRegisters.POSITIONING_ERROR_YZ]

        self.ang_vel = pypozyx.AngularVelocity()
        self.accel = pypozyx.Acceleration()

        self.posePub = rospy.Publisher('/uwb/odom', PoseWithCovarianceStamped, queue_size=10)
        self.imuPub = rospy.Publisher('/uwb/imu', Imu, queue_size=10)

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

        rospy.loginfo_throttle(5, good_status_msg)
        rospy.logwarn_throttle(5, bad_status_msg)

        if(status_all < 4):
            rospy.logwarn_throttle(5, 'Not enough anchors in range for 3D positioning (' + str(status_all) + '/4).')
            self.anchors_ready = False
        else:
            rospy.loginfo_throttle(60, status_all + ' anchors are in range.')
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
        cov_success = 1
        for register, dat in zip(self.cov_registers, self.cov):
            cov_success &= self.tag.regRead(register, dat)
        cov_matrix = np.array([[self.cov[0][0]/(1000**2), self.cov[3][0]/(1000**2), self.cov[4][0]/(1000**2), 0, 0, 0],
                               [self.cov[3][0]/(1000**2), self.cov[1][0]/(1000**2), self.cov[5][0]/(1000**2), 0, 0, 0],
                               [self.cov[4][0]/(1000**2), self.cov[5][0]/(1000**2), self.cov[2][0]/(1000**2), 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])

        if((pos_success & rot_success) & cov_success):
            msgPos = PoseWithCovarianceStamped()
            msgPos.header.stamp = rospy.Time.now()
            msgPos.header.frame_id = 'odom' # TODO fix

            msgPos.position.x = self.position.x/1000
            msgPos.position.y = self.position.y/1000
            msgPos.position.z = self.position.z/1000
            msgPos.orientation.x = self.quat.x
            msgPos.orientation.y = self.quat.y
            msgPos.orientation.z = self.quat.z
            msgPos.orientation.w = self.quat.w
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
            msgIMU.header.frame_id = 'odom' # TODO fix

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

    def calibrate_variance(self):
        return

if __name__ == '__main__':
    rospy.init_node('uwb_odom')

    node = uwbOdom()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if(not node.tag_ready):
            node.connectTag()
        elif(not node.anchors_added):
            node.connectAnchors()
        else:
            node.checkAnchorStatus()
            node.publishUWB()

        # if(node.anchors_ready):
        #     node.publishUWB()
        # elif(not node.tag_ready):
        #     node.connectTag()
        # elif(not node.anchors_added):
        #     node.connectAnchors()
        # else:
        #     node.checkAnchorStatus()

        rate.sleep()