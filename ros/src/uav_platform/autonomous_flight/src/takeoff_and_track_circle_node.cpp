/*
	FILE: takeoff_and_track_circle_node.cpp
	---------------------------
	Simple flight test for autonomous flight
*/

#include <ros/ros.h>
#include <autonomous_flight/flightBase.h>

int main(int argc, char** argv){
	ros::init(argc, argv, "takeoff_and_track_circle_node");
	ros::NodeHandle nh;

	AutoFlight::flightBase fb (nh);
	fb.takeoff();
	fb.run();
	ros::spin();
	return 0;
}