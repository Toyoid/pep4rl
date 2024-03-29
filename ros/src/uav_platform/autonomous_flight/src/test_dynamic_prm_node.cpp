/*
	FILE: .cpp
	-----------------------------
	ROS node for testing incremental construction of probabilistic roadmap
*/

#include <autonomous_flight/dynamicPRM.h>

int main(int argc, char** argv){
	ros::init(argc, argv, "dynamic_prm_node");
	ros::NodeHandle nh;
	AutoFlight::dynamicPRM d_prm (nh);
	d_prm.run();

	ros::spin();

	return 0;
}